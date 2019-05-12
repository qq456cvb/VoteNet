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
    poly1_xz = Polygon(np.stack([bbox1[:4, 0], bbox1[:4, 2]], -1))
    poly2_xz = Polygon(np.stack([bbox2[:4, 0], bbox2[:4, 2]], -1))
    iou_xz = poly1_xz.intersection(poly2_xz).area / poly1_xz.union(poly2_xz).area
    return max(iou_xz * min(bbox1[0, 1], bbox2[0, 1]) - max(bbox1[4, 1], bbox2[4, 1]), 0)


"""
 Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
         precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.
"""
def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


# idea reference: https://github.com/Cartucho/mAP
def eval_mAP(dataset, pred_func, ious):
    fps = {iou: {t: [] for t in type2class} for iou in ious}
    tps = {iou: {t: [] for t in type2class} for iou in ious}
    confidence = {t: [] for t in type2class}
    aps = {iou: {t: 0 for t in type2class} for iou in ious}
    gt_counter_per_class = {t: 0 for t in type2class}
    for idx in range(len(dataset)):
        try:
            calib = dataset.get_calibration(idx)
            objects = dataset.get_label_objects(idx)
            pc_upright_depth = dataset.get_depth(idx)
            pc_upright_depth = pc_upright_depth[
                                np.random.choice(pc_upright_depth.shape[0], config.POINT_NUM, replace=False), :]  # subsample
            pc_upright_camera = np.zeros_like(pc_upright_depth)
            pc_upright_camera[:, 0:3] = calib.project_upright_depth_to_upright_camera(pc_upright_depth[:, 0:3])
            pc_upright_camera[:, 3:] = pc_upright_depth[:, 3:]
            pc_image_coord, _ = calib.project_upright_depth_to_image(pc_upright_depth)

            bboxes_pred, class_scores_pred, _ = pred_func(pc_upright_camera[None, :, :3])
            class_score_pred = np.max(class_scores_pred, axis=-1)
            # sort by confidence, high 2 low
            # sort_idx = np.argsort(-np.max(class_scores_pred, axis=-1))
            # bboxes_pred = bboxes_pred[sort_idx]
            # class_scores_pred = class_scores_pred[sort_idx]
            class_labels_pred = np.argmax(class_scores_pred, -1)

            if not objects:
                continue

            gt_bboxes = []
            gt_classes = []
            for obj_idx in range(len(objects)):
                obj = objects[obj_idx]
                if obj.classname not in type_whitelist:
                    continue

                # 2D BOX: Get pts rect backprojected
                box2d = obj.box2d
                xmin, ymin, xmax, ymax = box2d
                box_fov_inds = (pc_image_coord[:, 0] < xmax) & (pc_image_coord[:, 0] >= xmin) & (
                        pc_image_coord[:, 1] < ymax) & (pc_image_coord[:, 1] >= ymin)
                pc_in_box_fov = pc_upright_camera[box_fov_inds, :]
                # Get frustum angle (according to center pixel in 2D BOX)
                # 3D BOX: Get pts velo in 3d box
                box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib)
                box3d_pts_3d = calib.project_upright_depth_to_upright_camera(box3d_pts_3d)
                _, inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
                # Get 3D BOX size
                box3d_size = np.array([2 * obj.l, 2 * obj.w, 2 * obj.h])
                box3d_center = (box3d_pts_3d[0, :] + box3d_pts_3d[6, :]) / 2

                # Size
                size_class, size_residual = size2class(box3d_size, obj.classname)
                angle_class, angle_residual = angle2class(obj.heading_angle, config.NH)

                # Reject object with too few points
                if len(inds) < 5:
                    continue

                bbox = get_3d_box(class2size(size_class, size_residual),
                           class2angle(angle_class, angle_residual, config.NH), box3d_center)
                gt_bboxes.append(bbox)
                gt_classes.append(type2class[obj.classname])
                gt_counter_per_class[obj.classname] += 1

            gt_matched = {k: False for k in range(len(gt_bboxes))}
            for i, bbox_pred in enumerate(bboxes_pred):
                max_overlap = -1
                gt_match = -1
                for j, gt_bbox in enumerate(gt_bboxes):
                    if gt_classes[j] == class_labels_pred[i]:
                        overlap = iou_3d(bbox_pred, gt_bbox)
                        if overlap > max_overlap:
                            max_overlap = overlap
                            gt_match = j

                for iou in ious:
                    if max_overlap > iou and not gt_matched[gt_match]:
                        gt_matched[gt_match] = True
                        tps[iou][class_labels_pred[i]].append(1)
                        fps[iou][class_labels_pred[i]].append(0)
                    else:
                        tps[iou][class_labels_pred[i]].append(0)
                        fps[iou][class_labels_pred[i]].append(1)

                confidence[class_labels_pred[i]].append(class_score_pred[i])

        except Exception as e:
            print(e)

    for iou in ious:
        for t in type2class:
            tp = tps[iou][t]
            fp = fps[iou][t]
            # sort by confidence
            tp.sort(key=lambda k: -confidence[tp.index(k)])
            fp.sort(key=lambda k: -confidence[fp.index(k)])
            tp = list(itertools.accumulate(tp))
            fp = list(itertools.accumulate(fp))

            rec = tp[:]
            for i, val in enumerate(tp):
                rec[i] = float(tp[i]) / gt_counter_per_class[t]
            # print(rec)
            prec = tp[:]
            for i, val in enumerate(tp):
                prec[i] = float(tp[i]) / (fp[i] + tp[i])
            # print(prec)

            ap, mrec, mprec = voc_ap(rec[:], prec[:])
            aps[iou][t] = ap

    return {iou: np.mean(list(aps[iou].values())) for iou in ious}


class Evaluator(Callback):
    def __init__(self, root, split, batch_size, idx_list=None):
        self.dataset = sunrgbd_object(root, split, idx_list)
        self.batch_size = batch_size  # not used for now

    def _setup_graph(self):
        self.pred_func = self.trainer.get_predictor(['points'], ['bboxes_pred', 'class_scores_pred', 'batch_idx'])

    def _before_train(self):
        mAPs = eval_mAP(self.dataset, self.pred_func, [0.25, 0.5])
        for iou in mAPs:
            logger.info(iou, " mAP:",  mAPs[iou])

    def _trigger_epoch(self):
        mAPs = eval_mAP(self.dataset, self.pred_func, [0.25, 0.5])
        for iou in mAPs:
            self.trainer.monitors.put_scalar('mAP%f' % iou, mAPs[iou])


if __name__ == '__main__':
    import itertools
    from model import Model
    print(eval_mAP(sunrgbd_object('/media/neil/DATA/mysunrgbd', 'training'), OfflinePredictor(PredictConfig(
            model=Model(),
            input_names=['points'],
            output_names=['bboxes_pred', 'class_scores_pred', 'batch_idx'])), [0.25]))