import tensorflow as tf
from tensorflow.python.framework import ops
from sunutils import roty
import sys
import os
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
nms3d = tf.load_op_library(os.path.join(BASE_DIR, 'tf_nms3d_so.so'))


def NMS3D(bboxes, scores, objectiveness, iou_threshold):
    return nms3d.non_max_suppression3d(bboxes, scores, objectiveness, iou_threshold)


if __name__=='__main__':
    import numpy as np
    import time
    # print(dir(nms3d))
    # print(help(nms3d.non_max_suppression3d))

    def bbox(l, w, h, roty_angle=None):
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        corners_3d = np.vstack([x_corners, y_corners, z_corners])
        if roty_angle:
            corners_3d = roty(roty_angle) @ corners_3d
        return np.transpose(corners_3d)
    np.random.seed(100)
    bboxes = np.array([[
        bbox(1, 1, 1),
        bbox(0.8, 0.8, 0.8, np.pi / 4 * 3) + np.array([[0, 0, 0]]),
    ]]).astype('float32')
    print(bboxes.shape)
    scores = np.array([[
        0.5, 0.6
    ]]).astype('float32')
    objectiveness = np.array([[
        [0.3, 0.7],
        [0.4, 0.6],
    ]]).astype('float32')
    with tf.device('/cpu:0'):
        bbox = tf.constant(bboxes)
        scores = tf.constant(scores)
        objectiveness = tf.constant(objectiveness)
        idx = nms3d.non_max_suppression3d(bbox, scores, objectiveness, 0.5)
    with tf.Session('') as sess:
        now = time.time()
        for _ in range(100):
            ret = sess.run(idx)
        # print(time.time() - now)
        print(ret)
        # print(ret.shape, ret.dtype)
