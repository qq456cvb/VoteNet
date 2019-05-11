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


# this is incorrect!!! TODO
def eval_mAP(dataset, pred_func, ious):
    gt_labels_all = {iou: {t: [] for t in type2class} for iou in ious}
    pred_scores_all = {t: [] for t in type2class}
    for idx in range(1, 10):
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
            # sort by confidence, high 2 low
            sort_idx = np.argsort(-np.max(class_scores_pred, axis=-1))
            bboxes_pred = bboxes_pred[sort_idx]
            class_scores_pred = class_scores_pred[sort_idx]
            class_labels_pred = np.argmax(class_scores_pred, -1)

            if not objects:
                continue

            gt_bboxes = []
            gt_classes = []
            gt_labels = {iou: [] for iou in ious}
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
                        gt_labels[iou].append(1)
                    else:
                        gt_labels[iou].append(0)

            if len(class_labels_pred) > 0:
                for iou in ious:
                    for t in type2class:
                        cls_idx = class_labels_pred == type2class[t]
                        pred_scores_all[t].extend(np.max(class_scores_pred, axis=-1)[cls_idx])
                        gt_labels_all[iou][t].extend(np.asarray(gt_labels[iou])[cls_idx])
        except Exception as e:
            print(e)

    aps = {iou: {t: 0 for t in type2class} for iou in ious}
    for iou in ious:
        for t in type2class:
            aps[iou][t] = average_precision_score(gt_labels_all[iou][t], pred_scores_all[t])

    return aps


# TODO: compute mAP
class Evaluator(Callback):
    def __init__(self, root, split, batch_size):
        self.dataset = sunrgbd_object(root, split)
        self.batch_size = batch_size  # not used for now

    def _setup_graph(self):
        self.pred_func = self.trainer.get_predictor(['points'], ['bboxes_pred', 'class_scores_pred', 'batch_idx'])

    def _before_train(self):
        print(eval_mAP(self.dataset, self.pred_func, [0.25]))

    def _trigger_epoch(self):
        pass
