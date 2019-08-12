''' Helper class and functions for loading SUN RGB-D objects

Author: Charles R. Qi
Date: October 2017

TODO: code formatting and clean-up.
'''

import os
import sys
import numpy as np
from mayavi import mlab
import config
from tensorpack import *
import sys
import glob
from timeit import default_timer as timer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import utils
from sunutils import *

from viz_utils import draw_gt_boxes3d, draw_lidar
import cv2
from PIL import Image

data_dir = BASE_DIR
AUGMENT_X = 5

type2class = {'bed': 0, 'table': 1, 'sofa': 2, 'chair': 3, 'toilet': 4, 'desk': 5, 'dresser': 6, 'night_stand': 7,
              'bookshelf': 8, 'bathtub': 9}
class2type = {type2class[t]: t for t in type2class}
type2onehotclass = {'bed': 0, 'table': 1, 'sofa': 2, 'chair': 3, 'toilet': 4, 'desk': 5, 'dresser': 6, 'night_stand': 7,
                    'bookshelf': 8, 'bathtub': 9}
type_mean_size = {'bathtub': np.array([0.765840, 1.398258, 0.472728]),
                  'bed': np.array([2.114256, 1.620300, 0.927272]),
                  'bookshelf': np.array([0.404671, 1.071108, 1.688889]),
                  'chair': np.array([0.591958, 0.552978, 0.827272]),
                  'desk': np.array([0.695190, 1.346299, 0.736364]),
                  'dresser': np.array([0.528526, 1.002642, 1.172878]),
                  'night_stand': np.array([0.500618, 0.632163, 0.683424]),
                  'sofa': np.array([0.923508, 1.867419, 0.845495]),
                  'table': np.array([0.791118, 1.279516, 0.718182]),
                  'toilet': np.array([0.699104, 0.454178, 0.756250])}

class_mean_size = np.zeros((len(type2class), 3), dtype=np.float32)
for t, idx in type2class.items():
    class_mean_size[idx] = type_mean_size[t]


def angle2class(angle, num_class):
    ''' Convert continuous angle to discrete class
        [optinal] also small regression number from
        class center angle to current angle.

        angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        return is class of int32 of 0,1,...,N-1 and a number such that
            class*(2pi/N) + number = angle
    '''
    angle = angle % (2 * np.pi)
    assert (angle >= 0 and angle <= 2 * np.pi)
    angle_per_class = 2 * np.pi / float(num_class)
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    class_id = int(shifted_angle / angle_per_class)
    residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
    return class_id, residual_angle


def class2angle(pred_cls, residual, num_class, to_label_format=True):
    ''' Inverse function to angle2class '''
    angle_per_class = 2 * np.pi / float(num_class)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle > np.pi:
        angle = angle - 2 * np.pi
    return angle


def size2class(size, type_name):
    ''' Convert 3D box size (l,w,h) to size class and size residual '''
    size_class = type2class[type_name]
    size_residual = size - type_mean_size[type_name]
    return size_class, size_residual


def class2size(pred_cls, residual):
    ''' Inverse function to size2class '''
    mean_size = type_mean_size[class2type[pred_cls]]
    return mean_size + residual


def get_3d_box(box_size, heading_angle, center):
    ''' box_size is array(l,w,h), heading_angle is radius clockwise from pos x axis, center is xyz of box center
        output (8,3) array for 3D box cornders
        Similar to utils/compute_orientation_3d
    '''
    R = roty(heading_angle)
    l, w, h = box_size
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d


class sunrgbd_object(object):
    ''' Load and parse object data '''

    def __init__(self, root_dir, split='training', idx_list=None):
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        # if split == 'training':
        #     self.num_samples = 10335
        # elif split == 'testing':
        #     self.num_samples = 2860
        # else:
        #     print('Unknown split: %s' % (split))
        #     exit(-1)
        self.samples = idx_list if idx_list is not None else list(range(1, 10336 if split == 'training' else 2861))

        self.image_dir = os.path.join(self.split_dir, 'image')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.depth_dir = os.path.join(self.split_dir, 'depth')
        self.label_dir = os.path.join(self.split_dir, 'label_dimension')
        # self.label_dimension_dir = os.path.join(self.split_dir, 'label_dimension')

    def __len__(self):
        return len(self.samples)

    def get_image(self, idx):
        img_filename = os.path.join(self.image_dir, '%06d.jpg' % (idx))
        return load_image(img_filename)

    def get_depth(self, idx):
        depth_filename = os.path.join(self.depth_dir, '%06d.txt' % (idx))
        return load_depth_points(depth_filename)

    def get_calibration(self, idx):
        calib_filename = os.path.join(self.calib_dir, '%06d.txt' % (idx))
        return SUNRGBD_Calibration(calib_filename)

    def get_label_objects(self, idx):
        # assert (self.split == 'training')
        label_filename = os.path.join(self.label_dir, '%06d.txt' % (idx))
        return read_sunrgbd_label(label_filename)


class MyDataFlow(RNGDataFlow):
    def __init__(self, root, split, training, idx_list=None, cache_dir=None):
        self.dataset = sunrgbd_object(root, split, idx_list)
        self.training = training
        self.type_whitelist = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser', 'night_stand',
                                    'bookshelf', 'bathtub')
        self.cache_dir = cache_dir
        if self.cache_dir:
            if not os.path.exists(self.cache_dir):
                os.mkdir(self.cache_dir)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        if self.training:
            self.rng.shuffle(self.dataset.samples)
        for idx in self.dataset.samples:
            objects = self.dataset.get_label_objects(idx)
            if not objects:
                continue

            if self.cache_dir is None:
                cache_cnt = 0
            else:
                cache_cnt = len(glob.glob(os.path.join(self.cache_dir, 'data%d_*.npy' % idx)))
            # augment each scene 5 times
            if cache_cnt < (AUGMENT_X if self.training else 1):
                calib = self.dataset.get_calibration(idx)
                pc_upright_depth = self.dataset.get_depth(idx)
                pc_upright_depth = pc_upright_depth[
                                    self.rng.choice(pc_upright_depth.shape[0], config.POINT_NUM, replace=False), :]  # subsample
                pc_upright_camera = np.zeros_like(pc_upright_depth)
                pc_upright_camera[:, 0:3] = calib.project_upright_depth_to_upright_camera(pc_upright_depth[:, 0:3])
                pc_upright_camera[:, 3:] = pc_upright_depth[:, 3:]
                pc_image_coord, _ = calib.project_upright_depth_to_image(pc_upright_depth)

            if self.training:
                if self.cache_dir is None:
                    augment = self.rng.randint(AUGMENT_X)
                else:
                    fns = glob.glob(os.path.join(self.cache_dir, 'data%d_*.npy' % idx))
                    exists = set([int(fn.split('_')[-1].split('.')[0]) for fn in fns])
                    cands = set(range(AUGMENT_X)) - exists
                    if not cands:
                        augment = self.rng.randint(AUGMENT_X)
                    else:
                        augment = list(cands)[0]
            else:
                augment = 0

            try:
                if self.cache_dir is None:
                    raise FileNotFoundError

                batch = pickle.load(
                    open(os.path.join(self.cache_dir, 'data%d_%d.npy' % (idx, augment)), 'rb'))
                if not batch:
                    continue
                yield batch
            except Exception as ex:
                if ex.__class__ not in [OSError, FileNotFoundError]:
                    pass

                if self.training:
                    if np.random.rand() > 0.5:
                        flip_x = True
                    else:
                        flip_x = False

                    if np.random.rand() > 0.5:
                        flip_z = True
                    else:
                        flip_z = False

                    rand_roty_angle = (np.random.rand() * 2 - 1.) * 5. / 180 * np.pi
                    rand_scale = (np.random.rand() * 2 - 1.) * 0.1 + 1.

                bboxes_xyz = []
                bboxes_lwh = []
                bboxes_roty = []
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
                    if np.max(box3d_pts_3d[:, 1]) - np.min(box3d_pts_3d[:, 1]) < 1e-7:   # SUNRGBD sometimes gives a degenerate bbox
                        continue
                    _, inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
                    # Get 3D BOX size
                    box3d_size = np.array([2 * obj.l, 2 * obj.w, 2 * obj.h])
                    box3d_center = (box3d_pts_3d[0, :] + box3d_pts_3d[6, :]) / 2

                    if self.training:
                        if flip_x:
                            box3d_center[..., 0] = -box3d_center[..., 0]
                            obj.heading_angle = np.pi - obj.heading_angle
                        if flip_z:
                            box3d_center[..., 2] = -box3d_center[..., 2]
                            obj.heading_angle = -obj.heading_angle

                        box3d_center = (roty(rand_roty_angle) @ box3d_center.T).T
                        obj.heading_angle += rand_roty_angle

                        box3d_center = box3d_center * rand_scale
                        box3d_size = box3d_size * rand_scale

                    # Size
                    size_class, size_residual = size2class(box3d_size, obj.classname)

                    angle_class, angle_residual = angle2class(obj.heading_angle, config.NH)

                    # Reject object with too few points
                    if len(inds) < 5:
                        continue

                    # VISUALIZE
                    # img2 = np.copy(self.dataset.get_image(idx))
                    # cv2.rectangle(img2, (int(obj.xmin), int(obj.ymin)), (int(obj.xmax), int(obj.ymax)), (0, 255, 0),
                    #               2)
                    # draw_projected_box3d(img2, box3d_pts_2d)
                    # Image.fromarray(img2).show()

                    bboxes_xyz.append(box3d_center)
                    bboxes_lwh.append(box3d_size)
                    bboxes_roty.append(obj.heading_angle)
                    semantic_labels.append(type2class[obj.classname])
                    heading_labels.append(angle_class)
                    heading_residuals.append(angle_residual / (np.pi / config.NH))
                    size_labels.append(size_class)
                    size_residuals.append(size_residual / type_mean_size[obj.classname])

                if len(bboxes_xyz) > 0:
                    if self.training:
                        if flip_x:
                            pc_upright_camera[..., 0] = -pc_upright_camera[..., 0]
                        if flip_z:
                            pc_upright_camera[..., 2] = -pc_upright_camera[..., 2]
                        pc_upright_camera[:, :3] = (roty(rand_roty_angle) @ pc_upright_camera[:, :3].T).T
                        pc_upright_camera[:, :3] = pc_upright_camera[:, :3] * rand_scale

                    batch = [idx, pc_upright_camera[:, :3], np.array(bboxes_xyz), np.array(bboxes_lwh), np.asarray(bboxes_roty), np.array(semantic_labels),
                           np.array(heading_labels), np.array(heading_residuals), np.array(size_labels), np.array(size_residuals)]
                    if self.cache_dir is not None:
                        with open(os.path.join(self.cache_dir, 'data%d_%d.npy' % (idx, augment)), 'wb') as f:
                            pickle.dump(batch, f)
                    yield batch
                else:
                    with open(os.path.join(self.cache_dir, 'data%d_%d.npy' % (idx, augment)), 'wb') as f:
                        pickle.dump([], f)  # dummy


if __name__ == '__main__':
    # dataset_viz()
    # get_box3d_dim_statistics('/home/rqi/Data/mysunrgbd/training/train_data_idx.txt')
    # extract_roi_seg('/home/rqi/Data/mysunrgbd/training/val_data_idx.txt', 'training',
    #                 output_filename='val_1002.zip.pickle', viz=False, augmentX=1)
    # extract_roi_seg('/home/rqi/Data/mysunrgbd/training/train_data_idx.txt', 'training',
    #                 output_filename='train.pickle', viz=False, augmentX=1)
    if __name__ == '__main__':
        import mayavi.mlab as mlab
        import config

        from viz_utils import draw_lidar, draw_gt_boxes3d

        median_list = []
        dataset = MyDataFlow('/data/mysunrgbd', 'training', training=True, idx_list=list(range(5051, 10336)), cache_dir=None)
        dataset.reset_state()
        # print(type(dataset.input_list[0][0, 0]))
        # print(dataset.input_list[0].shape)
        # print(dataset.input_list[2].shape)
        # input()
        for obj in dataset:
            for i in range(len(obj[2])):
                data = [o[i] for o in obj[1:]]
                print('Center: ', data[1], 'angle_class: ', data[4], 'angle_res:', data[5], 'size_class: ', data[6],
                      'size_residual:', data[7], 'real_size:', type_mean_size[class2type[data[6]]] + data[7])
                box3d_from_label = get_3d_box(class2size(data[6], data[7] * type_mean_size[class2type[data[6]]]), class2angle(data[4], data[5] * np.pi / config.NH, config.NH), data[1])
                # raw_input()
                print(box3d_from_label)
            break

                ## Recover original labels
                # rot_angle = dataset.get_center_view_rot_angle(i)
                # print dataset.id_list[i]
                # print from_prediction_to_label_format(data[2], data[3], data[4], data[5], data[6], rot_angle)

                # ps = obj[0]
                # fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4), fgcolor=None, engine=None, size=(1000, 500))
                # mlab.points3d(ps[:, 0], ps[:, 1], ps[:, 2], mode='point', colormap='gnuplot', scale_factor=1,
                #               figure=fig)
                # mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2, figure=fig)
                # # draw_gt_boxes3d([dataset.get_center_view_box3d(i)], fig)
                # draw_gt_boxes3d([box3d_from_label], fig, color=(1, 0, 0))
                # mlab.orientation_axes()
                # print(ps[0:10, :])
                # mlab.show()
    # extract_roi_seg_from_rgb_detection('FPN_384x384', 'training', 'fcn_det_val.zip.pickle', valid_id_list=[int(line.rstrip()) for line in open('/home/rqi/Data/mysunrgbd/training/val_data_idx.txt')], viz=True)