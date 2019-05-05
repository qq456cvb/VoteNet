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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import utils
from sunutils import *

from viz_utils import draw_gt_boxes3d, draw_lidar
import cv2
from PIL import Image

data_dir = BASE_DIR

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


def rotate_pc_along_y(pc, rot_angle):
    ''' Input ps is NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc


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
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2];
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2];
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2];
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + center[0];
    corners_3d[1, :] = corners_3d[1, :] + center[1];
    corners_3d[2, :] = corners_3d[2, :] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d


class sunrgbd_object(object):
    ''' Load and parse object data '''

    def __init__(self, root_dir, split='training'):
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        if split == 'training':
            self.num_samples = 10335
        elif split == 'testing':
            self.num_samples = 2860
        else:
            print('Unknown split: %s' % (split))
            exit(-1)

        self.image_dir = os.path.join(self.split_dir, 'image')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.depth_dir = os.path.join(self.split_dir, 'depth')
        self.label_dir = os.path.join(self.split_dir, 'label_dimension')
        # self.label_dimension_dir = os.path.join(self.split_dir, 'label_dimension')

    def __len__(self):
        return self.num_samples

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
        assert (self.split == 'training')
        label_filename = os.path.join(self.label_dir, '%06d.txt' % (idx))
        return read_sunrgbd_label(label_filename)


def dataset_viz(show_frustum=False):
    sunrgbd = sunrgbd_object(data_dir)
    idxs = np.array(range(1, len(sunrgbd) + 1))
    np.random.shuffle(idxs)
    for idx in range(len(sunrgbd)):
        data_idx = idxs[idx]
        print('--------------------', data_idx)
        pc = sunrgbd.get_depth(data_idx)
        print(pc.shape)

        # Project points to image
        calib = sunrgbd.get_calibration(data_idx)
        uv, d = calib.project_upright_depth_to_image(pc[:, 0:3])
        print(uv)
        print(d)
        input()

        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap('hsv', 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

        img = sunrgbd.get_image(data_idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(uv.shape[0]):
            depth = d[i]
            color = cmap[int(120.0 / depth), :]
            cv2.circle(img, (int(np.round(uv[i, 0])), int(np.round(uv[i, 1]))), 2, color=tuple(color), thickness=-1)
        Image.fromarray(img).show()
        input()

        # Load box labels
        objects = sunrgbd.get_label_objects(data_idx)
        print(objects)
        input()

        # Draw 2D boxes on image
        img = sunrgbd.get_image(data_idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i, obj in enumerate(objects):
            cv2.rectangle(img, (int(obj.xmin), int(obj.ymin)), (int(obj.xmax), int(obj.ymax)), (0, 255, 0), 2)
            cv2.putText(img, '%d %s' % (i, obj.classname), (max(int(obj.xmin), 15), max(int(obj.ymin), 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        Image.fromarray(img).show()
        input()

        # Draw 3D boxes on depth points
        box3d = []
        ori3d = []
        for obj in objects:
            corners_3d_image, corners_3d = utils.compute_box_3d(obj, calib)
            ori_3d_image, ori_3d = utils.compute_orientation_3d(obj, calib)
            print('Corners 3D: ', corners_3d)
            box3d.append(corners_3d)
            ori3d.append(ori_3d)
        input()

        bgcolor = (0, 0, 0)
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000))
        mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 2], mode='point', colormap='gnuplot', figure=fig)
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)
        draw_gt_boxes3d(box3d, fig=fig)
        for i in range(len(ori3d)):
            ori_3d = ori3d[i]
            x1, y1, z1 = ori_3d[0, :]
            x2, y2, z2 = ori_3d[1, :]
            mlab.plot3d([x1, x2], [y1, y2], [z1, z2], color=(0.5, 0.5, 0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.orientation_axes()
        for i, obj in enumerate(objects):
            print('Orientation: ', i, np.arctan2(obj.orientation[1], obj.orientation[0]))
            print('Dimension: ', i, obj.l, obj.w, obj.h)
        input()

        if show_frustum:
            img = sunrgbd.get_image(data_idx)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for i, obj in enumerate(objects):
                box2d_fov_inds = (uv[:, 0] < obj.xmax) & (uv[:, 0] >= obj.xmin) & (uv[:, 1] < obj.ymax) & (
                            uv[:, 1] >= obj.ymin)
                box2d_fov_pc = pc[box2d_fov_inds, :]
                img2 = np.copy(img)
                cv2.rectangle(img2, (int(obj.xmin), int(obj.ymin)), (int(obj.xmax), int(obj.ymax)), (0, 255, 0), 2)
                cv2.putText(img2, '%d %s' % (i, obj.classname), (max(int(obj.xmin), 15), max(int(obj.ymin), 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                Image.fromarray(img2).show()

                fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1000, 1000))
                mlab.points3d(box2d_fov_pc[:, 0], box2d_fov_pc[:, 1], box2d_fov_pc[:, 2], box2d_fov_pc[:, 2],
                              mode='point', colormap='gnuplot', figure=fig)
                input()


def extract_roi_seg(idx_filename, split, output_filename, viz, perturb_box2d=False, augmentX=1,
                    type_whitelist=('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser', 'night_stand',
                                    'bookshelf', 'bathtub')):
    dataset = sunrgbd_object('/media/neil/DATA/mysunrgbd', split)
    # data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    data_idx_list = list(range(1, 10335))

    final_list = []

    pos_cnt = 0
    all_cnt = 0
    for data_idx in data_idx_list:
        try:
            print('------------- ', data_idx)
            calib = dataset.get_calibration(data_idx)
            objects = dataset.get_label_objects(data_idx)
            pc_upright_depth = dataset.get_depth(data_idx)
            pc_upright_camera = np.zeros_like(pc_upright_depth)
            pc_upright_camera[:, 0:3] = calib.project_upright_depth_to_upright_camera(pc_upright_depth[:, 0:3])
            pc_upright_camera[:, 3:] = pc_upright_depth[:, 3:]
            if viz:
                mlab.points3d(pc_upright_camera[:, 0], pc_upright_camera[:, 1], pc_upright_camera[:, 2],
                              pc_upright_camera[:, 1], mode='point')
                mlab.orientation_axes()
                mlab.show()
            img = dataset.get_image(data_idx)
            img_height, img_width, img_channel = img.shape
            pc_image_coord, _ = calib.project_upright_depth_to_image(pc_upright_depth)
        except:
            continue
        # print('PC image coord: ', pc_image_coord)

        # box2d_list = []  # [xmin,ymin,xmax,ymax]
        # box3d_list = []  # (8,3) array in upright depth coord
        input_list = []  # channel number = 6, xyz,rgb in upright depth coord
        label_list = []  # 1 for roi object, 0 for clutter
        # type_list = []  # string e.g. bed
        # heading_list = []  # face of object angle, radius of clockwise angle from positive x axis in upright camera coord
        # box3d_size_list = []  # array of l,w,h
        # frustum_angle_list = []  # angle of 2d box center from pos x-axis (clockwise)
        obj_list = []

        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            if obj.classname not in type_whitelist: continue

            # 2D BOX: Get pts rect backprojected
            box2d = obj.box2d
            for _ in range(augmentX):
                try:
                    # Augment data by box2d perturbation
                    if perturb_box2d:
                        xmin, ymin, xmax, ymax = random_shift_box2d(box2d)
                        print(xmin, ymin, xmax, ymax)
                    else:
                        xmin, ymin, xmax, ymax = box2d
                    box_fov_inds = (pc_image_coord[:, 0] < xmax) & (pc_image_coord[:, 0] >= xmin) & (
                                pc_image_coord[:, 1] < ymax) & (pc_image_coord[:, 1] >= ymin)
                    pc_in_box_fov = pc_upright_camera[box_fov_inds, :]
                    # Get frustum angle (according to center pixel in 2D BOX)
                    box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
                    uvdepth = np.zeros((1, 3))
                    uvdepth[0, 0:2] = box2d_center
                    uvdepth[0, 2] = 20  # some random depth
                    box2d_center_upright_camera = calib.project_image_to_upright_camerea(uvdepth)
                    print('UVdepth, center in upright camera: ', uvdepth, box2d_center_upright_camera)
                    frustum_angle = -1 * np.arctan2(box2d_center_upright_camera[0, 2], box2d_center_upright_camera[
                        0, 0])  # angle as to positive x-axis as in the Zoox paper
                    print('Frustum angle: ', frustum_angle)
                    # 3D BOX: Get pts velo in 3d box
                    box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib)
                    box3d_pts_3d = calib.project_upright_depth_to_upright_camera(box3d_pts_3d)
                    _, inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
                    print(len(inds))
                    label = np.zeros((pc_in_box_fov.shape[0]))
                    label[inds] = 1
                    # Get 3D BOX heading
                    print('Orientation: ', obj.orientation)
                    print('Heading angle: ', obj.heading_angle)
                    # Get 3D BOX size
                    box3d_size = np.array([2 * obj.l, 2 * obj.w, 2 * obj.h])
                    print('Box3d size: ', box3d_size)
                    print('Type: ', obj.classname)
                    print('Num of point: ', pc_in_box_fov.shape[0])

                    # Subsample points..
                    num_point = pc_in_box_fov.shape[0]
                    if num_point > 2048:
                        choice = np.random.choice(pc_in_box_fov.shape[0], 2048, replace=False)
                        pc_in_box_fov = pc_in_box_fov[choice, :]
                        label = label[choice]
                    # Reject object with too few points
                    if np.sum(label) < 5:
                        continue

                    # id_list.append(data_idx)
                    # box2d_list.append(np.array([xmin, ymin, xmax, ymax]))
                    # box3d_list.append(box3d_pts_3d)
                    input_list.append(pc_in_box_fov)
                    label_list.append(label)
                    # type_list.append(obj.classname)
                    # heading_list.append(obj.heading_angle)
                    # box3d_size_list.append(box3d_size)
                    # frustum_angle_list.append(frustum_angle)
                    obj_list.append([box3d_pts_3d, obj.classname, obj.heading_angle, box3d_size])


                    # collect statistics
                    pos_cnt += np.sum(label)
                    all_cnt += pc_in_box_fov.shape[0]

                    # VISUALIZATION
                    if viz:
                        img2 = np.copy(img)
                        cv2.rectangle(img2, (int(obj.xmin), int(obj.ymin)), (int(obj.xmax), int(obj.ymax)), (0, 255, 0),
                                      2)
                        draw_projected_box3d(img2, box3d_pts_2d)
                        Image.fromarray(img2).show()
                        p1 = input_list[-1]
                        seg = label_list[-1]
                        fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4), fgcolor=None, engine=None,
                                          size=(1024, 1024))
                        mlab.points3d(p1[:, 0], p1[:, 1], p1[:, 2], seg, mode='point', colormap='gnuplot',
                                      scale_factor=1, figure=fig)
                        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)
                        draw_gt_boxes3d([box3d_pts_3d], fig=fig)
                        mlab.orientation_axes()
                        mlab.show()
                except Exception as e:
                    print(e)
        if obj_list:
            print(pc_upright_camera.shape, len(obj_list))
            final_list.append([data_idx, pc_upright_camera, obj_list])

    save_zipped_pickle(final_list, output_filename)
    # print('Average pos ratio: ', pos_cnt / float(all_cnt))
    # print('Average npoints: ', float(all_cnt) / len(id_list))

    # utils.save_zipped_pickle(
    #     [id_list, box2d_list, box3d_list, input_list, label_list, type_list, heading_list, box3d_size_list,
    #      frustum_angle_list], output_filename)


class MyDataFlow(RNGDataFlow):
    def __init__(self, root, split):
        self.dataset = sunrgbd_object(root, split)
        self.training = split == 'training'
        self.type_whitelist = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser', 'night_stand',
                                    'bookshelf', 'bathtub')

    def __len__(self):
        return 10335 if self.training else 2860

    def __iter__(self):
        while True:
            try:
                idx = self.rng.randint(len(self))
                calib = self.dataset.get_calibration(idx)
                objects = self.dataset.get_label_objects(idx)
                pc_upright_depth = self.dataset.get_depth(idx)
                pc_upright_depth = pc_upright_depth[
                                    self.rng.choice(pc_upright_depth.shape[0], config.POINT_NUM, replace=False), :]  # subsample
                pc_upright_camera = np.zeros_like(pc_upright_depth)
                pc_upright_camera[:, 0:3] = calib.project_upright_depth_to_upright_camera(pc_upright_depth[:, 0:3])
                pc_upright_camera[:, 3:] = pc_upright_depth[:, 3:]
                pc_image_coord, _ = calib.project_upright_depth_to_image(pc_upright_depth)

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

                    # Data Augmentation: TODO

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
                    semantic_labels.append(type2class[obj.classname])
                    heading_labels.append(angle_class)
                    heading_residuals.append(angle_residual / (np.pi / config.NH))
                    size_labels.append(size_class)
                    size_residuals.append(size_residual / type_mean_size[obj.classname])

                if len(bboxes_xyz) > 0:
                    yield [pc_upright_camera[:, :3], np.array(bboxes_xyz), np.array(bboxes_lwh), np.array(semantic_labels),
                           np.array(heading_labels), np.array(heading_residuals), np.array(size_labels), np.array(size_residuals)]
            except Exception as ex:
                print(ex)


def get_box3d_dim_statistics(idx_filename, type_whitelist=['bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
                                                           'night_stand', 'bookshelf', 'bathtub']):
    dataset = sunrgbd_object('/home/rqi/Data/mysunrgbd')
    dimension_list = []
    type_list = []
    ry_list = []
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx)  # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            if obj.classname not in type_whitelist: continue
            heading_angle = -1 * np.arctan2(obj.orientation[1], obj.orientation[0])
            dimension_list.append(np.array([obj.l, obj.w, obj.h]))
            type_list.append(obj.classname)
            ry_list.append(heading_angle)

    import pickle
    with open('box3d_dimensions.pickle', 'wb') as fp:
        pickle.dump(type_list, fp)
        pickle.dump(dimension_list, fp)
        pickle.dump(ry_list, fp)


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

        sys.path.append(os.path.join(BASE_DIR, '../../mayavi'))
        from viz_utils import draw_lidar, draw_gt_boxes3d

        median_list = []
        dataset = MyDataFlow('/media/neil/DATA/mysunrgbd', 'training')
        dataset.reset_state()
        # print(type(dataset.input_list[0][0, 0]))
        # print(dataset.input_list[0].shape)
        # print(dataset.input_list[2].shape)
        # input()
        for obj in dataset:
            for i in range(len(obj[1])):
                data = [o[i] for o in obj]
                print('Center: ', data[1], 'angle_class: ', data[4], 'angle_res:', data[5], 'size_class: ', data[6],
                      'size_residual:', data[7], 'real_size:', type_mean_size[class2type[data[6]]] + data[7])
                box3d_from_label = get_3d_box(class2size(data[6], data[7] * type_mean_size[class2type[data[6]]]), class2angle(data[4], data[5] * np.pi / config.NH, config.NH), data[1])
                # raw_input()

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