import random
import time
import multiprocessing
from tqdm import tqdm
from six.moves import queue

from tensorpack.utils.concurrency import StoppableThread, ShareSessionThread
from tensorpack.callbacks import Callback
from tensorpack.utils import logger
from dataset import *
import numpy as np
import config
from shapely.geometry import Polygon
import itertools
from sklearn.metrics import average_precision_score
from tensorpack.utils.stats import StatCounter
from tensorpack.utils.utils import get_tqdm_kwargs
from tqdm import tqdm
type_whitelist = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser', 'night_stand',
                               'bookshelf', 'bathtub')

import sys
import os


def iou_3d(bbox1, bbox2):
    '''
    return iou of two 3d bbox
    :param bbox1: 8 * 3
    :param bbox2: 8 * 3
    :return: float
    '''
    assert bbox1[0, 1] > bbox1[4, 1] and bbox2[0, 1] > bbox2[4, 1]
    poly1_xz = Polygon(np.stack([bbox1[:4, 0], bbox1[:4, 2]], -1))
    poly2_xz = Polygon(np.stack([bbox2[:4, 0], bbox2[:4, 2]], -1))
    inter_area = poly1_xz.intersection(poly2_xz).area
    inter_vol = inter_area * max(0.0, min(bbox1[0, 1], bbox2[0, 1]) - max(bbox1[4, 1], bbox2[4, 1]))
    iou = inter_vol / (poly1_xz.area * (bbox1[0, 1] - bbox1[4, 1]) + poly2_xz.area * (bbox2[0, 1] - bbox2[4, 1]) - inter_vol)
    return iou


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def eval_det_cls(pred, gt, ovthresh=0.25):
    """ Generic functions to compute precision/recall for object detection
        for a single class.
        Input:
            pred: map of {img_id: [(bbox, score)]} where bbox is numpy array
            gt: map of {img_id: [bbox]}
            ovthresh: scalar, iou threshold
            use_07_metric: bool, if True use VOC07 11 point method
        Output:
            rec: numpy array of length nd
            prec: numpy array of length nd
            ap: scalar, average precision
    """

    # construct gt objects
    class_recs = {}  # {img_id: {'bbox': bbox list, 'det': matched list}}
    npos = 0
    for img_id in gt.keys():
        bbox = np.array(gt[img_id])
        det = [False] * len(bbox)
        npos += len(bbox)
        class_recs[img_id] = {'bbox': bbox, 'det': det}
    # pad empty list to all other imgids
    for img_id in pred.keys():
        if img_id not in gt:
            class_recs[img_id] = {'bbox': np.array([]), 'det': []}

    # construct dets
    image_ids = []
    confidence = []
    BB = []
    for img_id in pred.keys():
        for box, score in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            BB.append(box)
    confidence = np.array(confidence)
    BB = np.array(BB)  # (nd,4 or 8,3)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, ...]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        if d % 100 == 0:
            print(d)
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            for j in range(BBGT.shape[0]):
                iou = iou_3d(bb, BBGT[j, ...])
                if iou > ovmax:
                    ovmax = iou
                    jmax = j

        # print d, ovmax
        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    logger.info('NPOS: ' + str(npos))
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)

    return rec, prec, ap


def eval_det(pred_all, gt_all, ovthresh=0.25, use_07_metric=False):
    """ Generic functions to compute precision/recall for object detection
        for multiple classes.
        Input:
            pred_all: map of {img_id: [(classname, bbox, score)]}
            gt_all: map of {img_id: [(classname, bbox)]}
            ovthresh: scalar, iou threshold
            use_07_metric: bool, if true use VOC07 11 point method
        Output:
            rec: {classname: rec}
            prec: {classname: prec_all}
            ap: {classname: scalar}
    """
    pred = {}  # map {classname: pred}
    gt = {}  # map {classname: gt}
    for img_id in pred_all.keys():
        for classname, bbox, score in pred_all[img_id]:
            if classname not in pred: pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            if classname not in gt: gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            pred[classname][img_id].append((bbox, score))
    for img_id in gt_all.keys():
        for classname, bbox in gt_all[img_id]:
            if classname not in gt: gt[classname] = {}
            if classname not in pred: pred[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            gt[classname][img_id].append(bbox)

    rec = {}
    prec = {}
    ap = {}
    for classname in gt.keys():
        logger.info('Computing AP for class: ' + classname)
        rec[classname], prec[classname], ap[classname] = eval_det_cls(pred[classname], gt[classname], ovthresh)
        logger.info(classname + ':' + str(ap[classname]))
        # logger.info(classname + ' rec:' + str(rec[classname]))

    return rec, prec, ap


class Evaluator(Callback):
    def __init__(self, df, gt_all):
        self.df = df
        self.df.reset_state()
        self.gt_all = gt_all

    def _setup_graph(self):
        self.pred_func = self.trainer.get_predictor(['points'], ['bboxes_pred', 'class_scores_pred'])

    def _trigger(self):
        logger.info('Evaluating mAP on validation set...')
        pred_all = {}
        for data in tqdm(self.df):

            bboxes_pred, class_scores_pred = self.pred_func(data[1][None, ...])
            # logger.info('detected num of bboxes: ' + str(bboxes_pred.shape[0]))
            max_idx = np.argmax(class_scores_pred, axis=-1)
            class_names = [class2type[i] for i in max_idx]
            scores = np.max(class_scores_pred, axis=-1)
            assert len(class_names) == len(bboxes_pred) == len(scores)

            # print(class_names, bboxes_pred, scores)

            pred_all[data[0]] = zip(class_names, list(bboxes_pred), scores)
        _, _, ap = eval_det(pred_all, self.gt_all)
        self.trainer.monitors.put_scalar('mAP0.25', np.mean([ap[classname] for classname in ap]))


if __name__ == '__main__':
    import itertools
    from model import Model
    mAPs = eval_mAP(sunrgbd_object('/home/neil/mysunrgbd', 'training', idx_list=list(range(11, 21))), OfflinePredictor(PredictConfig(
            model=Model(),
            session_init=SaverRestore('./train_log/run/checkpoint'),
            input_names=['points'],
            output_names=['bboxes_pred', 'class_scores_pred', 'batch_idx'])), [0.25, 0.5])
    for iou in mAPs:
        print("mAP{:.2f}:{:.4f}".format(iou, mAPs[iou]))
