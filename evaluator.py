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
from tensorpack.utils.stats import StatCounter
from tensorpack.utils.utils import get_tqdm_kwargs

import sys
import os


# TODO: compute mAP
class Evaluator(Callback):
    def eval_epoch(self):
        for idx in range(1000):
            try:
                calib = self.dataset.get_calibration(idx)
                objects = self.dataset.get_label_objects(idx)
                pc_upright_depth = self.dataset.get_depth(idx)
                pc_upright_depth = pc_upright_depth[
                                    self.rng.choice(pc_upright_depth.shape[0], config.POINT_NUM, replace=False), :]  # subsample
                pc_upright_camera = np.zeros_like(pc_upright_depth)
                pc_upright_camera[:, 0:3] = calib.project_upright_depth_to_upright_camera(pc_upright_depth[:, 0:3])
                pc_upright_camera[:, 3:] = pc_upright_depth[:, 3:]
                pc_image_coord, _ = calib.project_upright_depth_to_image(pc_upright_depth)

                bboxes_pred, classes_pred, batch_idx = self.pred_func([pc_upright_camera[None, :, :3]])

                if not objects:
                    continue

                bboxes_xyz = []
                bboxes_lwh = []
                semantic_labels = []
                heading_labels = []
                heading_residuals = []
                size_labels = []
                size_residuals = []
                for obj_idx in range(len(objects)):
                    obj = objects[obj_idx]
                    if obj.classname not in self.type_whitelist:
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

                    bboxes_xyz.append(box3d_center)
                    bboxes_lwh.append(box3d_size)
                    semantic_labels.append(type2class[obj.classname])
                    heading_labels.append(angle_class)
                    heading_residuals.append(angle_residual / (np.pi / config.NH))
                    size_labels.append(size_class)
                    size_residuals.append(size_residual / type_mean_size[obj.classname])

            except:
                pass

    def __init__(self, root, split, batch_size):
        self.dataset = sunrgbd_object(root, split)
        self.batch_size = batch_size

    def _setup_graph(self):
        self.pred_func = self.trainer.get_predictor(['points'], ['bboxes_pred', 'classes_pred', 'batch_idx'])

    def _before_train(self):
        logger.info("farmer win rate: {}".format(farmer_win_rate))
        logger.info("lord win rate: {}".format(1 - farmer_win_rate))
        # self.lord_win_rate.load(1 - farmer_win_rate)
        # if t > 10 * 60:  # eval takes too long
        #     self.eval_episode = int(self.eval_episode * 0.94)

    def _trigger_epoch(self):
        t = time.time()
        farmer_win_rate = eval_with_funcs(
            self.pred_funcs, self.eval_episode, self.get_player_fn, verbose=False)
        t = time.time() - t
        if t > 10 * 60:  # eval takes too long
            self.eval_episode = int(self.eval_episode * 0.94)
        self.trainer.monitors.put_scalar('farmer_win_rate', farmer_win_rate)
        self.trainer.monitors.put_scalar('lord_win_rate', 1 - farmer_win_rate)
