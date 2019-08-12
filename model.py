import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_nms'))
import tensorflow as tf
from tensorpack import *
import numpy as np
from tensorpack.tfutils import get_current_tower_context, gradproc, optimizer, summary, varreplace
from utils import pointnet_sa_module, pointnet_fp_module
from dataset import class_mean_size
from tf_nms3d import NMS3D
import config


class Model(ModelDesc):
    def inputs(self):
        return [
                tf.placeholder(tf.int32, [None,], 'data_idx'),
                tf.placeholder(tf.float32, [None, config.POINT_NUM , 3], 'points'),
                tf.placeholder(tf.float32, [None, None, 3], 'bboxes_xyz'),
                tf.placeholder(tf.float32, [None, None, 3], 'bboxes_lwh'),
                tf.placeholder(tf.float32, [None, None], 'bboxes_roty'),
                tf.placeholder(tf.int32, (None, None), 'semantic_labels_input'),
                tf.placeholder(tf.int32, (None, None), 'heading_labels_input'),
                tf.placeholder(tf.float32, (None, None), 'heading_residuals_input'),
                tf.placeholder(tf.int32, (None, None), 'size_labels_input'),
                tf.placeholder(tf.float32, (None, None, 3), 'size_residuals_input'),
                ]

    def build_graph(self, _, x, bboxes_xyz, bboxes_lwh, bboxes_roty, semantic_labels, heading_labels, heading_residuals, size_labels, size_residuals):
        l0_xyz = x
        l0_points = x

        # Set Abstraction layers
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=2048, radius=0.2, nsample=64,
                                                           mlp=[64, 64, 128], mlp2=None, group_all=False, scope='sa1')
        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=1024, radius=0.4, nsample=64,
                                                           mlp=[128, 128, 256], mlp2=None, group_all=False, scope='sa2')
        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=512, radius=0.8, nsample=64,
                                                           mlp=[128, 128, 256], mlp2=None, group_all=False, scope='sa3')
        l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=256, radius=1.2, nsample=64,
                                                           mlp=[128, 128, 256], mlp2=None, group_all=False, scope='sa4')
        # Feature Propagation layers
        l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256, 256], scope='fp1')
        seeds_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256, 256], scope='fp2')
        seeds_xyz = l2_xyz

        # Voting Module layers
        offset = tf.reshape(tf.concat([seeds_xyz, seeds_points], 2), [-1, 256 + 3])
        units = [256, 256, 256 + 3]
        for i in range(len(units)):
            offset = FullyConnected('voting%d' % i, offset, units[i], activation=BNReLU if i < len(units) - 1 else None)
        offset = tf.reshape(offset, [-1, 1024, 256 + 3])

        # B * N * 3
        votes = tf.concat([seeds_xyz, seeds_points], 2) + offset
        votes_xyz = votes[:, :, :3]
        dist2center = tf.abs(tf.expand_dims(seeds_xyz, 2) - tf.expand_dims(bboxes_xyz, 1))

        def rotate_pc_along_y(pc, rot_angle):
            batch_size = tf.shape(rot_angle)[0]
            c = tf.cos(rot_angle)
            s = tf.sin(rot_angle)
            zeros = tf.zeros_like(c)
            ones = tf.ones_like(c)
            rotation = tf.reshape(tf.stack([c, zeros, s, zeros, ones, zeros, -s, zeros, c], -1),
                                  tf.stack([batch_size, -1, 3, 3]))  # B * BB * 3 * 3
            return tf.einsum('ijkl,imjl->imjk', rotation, pc)

        dist2center = rotate_pc_along_y(dist2center, -bboxes_roty)  # rotate point clouds to align with bboxes
        surface_ind = tf.less(dist2center, tf.expand_dims(bboxes_lwh, 1) / 2.)  # B * N * BB * 3, bool
        surface_ind = tf.equal(tf.count_nonzero(surface_ind, -1), 3)  # B * N * BB
        surface_ind = tf.greater_equal(tf.count_nonzero(surface_ind, -1), 1)  # B * N, should be in at least one bbox

        dist2center_norm = tf.norm(dist2center, axis=-1)  # B * N * BB
        votes_assignment = tf.argmin(dist2center_norm, -1, output_type=tf.int32)  # B * N, int
        bboxes_xyz_votes_gt = tf.gather_nd(bboxes_xyz, tf.stack([
            tf.tile(tf.expand_dims(tf.range(tf.shape(votes_assignment)[0]), -1), [1, tf.shape(votes_assignment)[1]]),
            votes_assignment], 2))  # B * N * 3
        vote_reg_loss = tf.reduce_mean(tf.norm(votes_xyz - bboxes_xyz_votes_gt, ord=1, axis=-1) * tf.cast(surface_ind, tf.float32), name='vote_reg_loss')
        votes_points = votes[:, :, 3:]

        # Proposal Module layers
        # Farthest point sampling on seeds
        proposals_xyz, proposals_output, _ = pointnet_sa_module(votes_xyz, votes_points, npoint=config.PROPOSAL_NUM,
                                                                radius=0.3, nsample=64, mlp=[128, 128, 128],
                                                                mlp2=[128, 128, 5+2 * config.NH+4 * config.NS+config.NC],
                                                                group_all=False, scope='proposal',
                                                                sample_xyz=seeds_xyz)

        obj_cls_score = tf.identity(proposals_output[..., :2], 'obj_scores')

        nms_iou = tf.get_variable('nms_iou', shape=[], initializer=tf.constant_initializer(0.25), trainable=False)
        if not get_current_tower_context().is_training:

            def get_3d_bbox(box_size, heading_angle, center):
                batch_size = tf.shape(heading_angle)[0]
                c = tf.cos(heading_angle)
                s = tf.sin(heading_angle)
                zeros = tf.zeros_like(c)
                ones = tf.ones_like(c)
                rotation = tf.reshape(tf.stack([c, zeros, s, zeros, ones, zeros, -s, zeros, c], -1), tf.stack([batch_size, -1, 3, 3]))
                l, w, h = box_size[..., 0], box_size[..., 1], box_size[..., 2]  # lwh(xzy) order!!!
                corners = tf.reshape(tf.stack([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2,
                                               h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2,
                                               w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], -1),
                                     tf.stack([batch_size, -1, 3, 8]))
                return tf.einsum('ijkl,ijlm->ijmk', rotation, corners) + tf.expand_dims(center, 2)  # B * N * 8 * 3

            class_mean_size_tf = tf.constant(class_mean_size)
            size_cls_pred = tf.argmax(proposals_output[..., 5 + 2 * config.NH: 5 + 2 * config.NH + config.NS], axis=-1)
            size_cls_pred_onehot = tf.one_hot(size_cls_pred, depth=config.NS, axis=-1)  # B * N * NS
            size_residual_pred = tf.reduce_sum(tf.expand_dims(size_cls_pred_onehot, -1)
                                               * tf.reshape(proposals_output[..., 5+2 * config.NH + config.NS:5+2 * config.NH + 4 * config.NS], (-1, config.PROPOSAL_NUM, config.NS, 3)), axis=2)
            size_pred = tf.gather_nd(class_mean_size_tf, tf.expand_dims(size_cls_pred, -1)) * tf.maximum(1 + size_residual_pred, 1e-6)  # B * N * 3: size
            # with tf.control_dependencies([tf.print(size_pred[0, 0, 2])]):
            center_pred = proposals_xyz + proposals_output[..., 2:5]  # B * N * 3
            heading_cls_pred = tf.argmax(proposals_output[..., 5:5+config.NH], axis=-1)
            heading_cls_pred_onehot = tf.one_hot(heading_cls_pred, depth=config.NH, axis=-1)
            heading_residual_pred = tf.reduce_sum(heading_cls_pred_onehot
                                                  * proposals_output[..., 5 + config.NH:5+2 * config.NH], axis=2)
            heading_pred = tf.floormod((tf.cast(heading_cls_pred, tf.float32) * 2 + heading_residual_pred) * np.pi / config.NH, 2 * np.pi)

            # with tf.control_dependencies([tf.print(size_residual_pred[0, :10, :]), tf.print(size_pred[0, :10, :])]):
            bboxes = get_3d_bbox(size_pred, heading_pred, center_pred)  # B * N * 8 * 3,  lhw(xyz) order!!!

            # bbox_corners = tf.concat([bboxes[:, :, 6, :], bboxes[:, :, 0, :]], axis=-1)  # B * N * 6,  lhw(xyz) order!!!
            # with tf.control_dependencies([tf.print(bboxes[0, 0])]):
            nms_idx = NMS3D(bboxes, tf.reduce_max(proposals_output[..., -config.NC:], axis=-1), proposals_output[..., :2], nms_iou)  # Nnms * 2

            bboxes_pred = tf.gather_nd(bboxes, nms_idx, name='bboxes_pred')  # Nnms * 8 * 3
            class_scores_pred = tf.gather_nd(proposals_output[..., -config.NC:], nms_idx, name='class_scores_pred')  # Nnms * C
            batch_idx = tf.identity(nms_idx[:, 0], name='batch_idx')  # Nnms, this is used to identify between batches

            return

        # calculate positive and negative proposal idxes
        bboxes_xyz_gt = bboxes_xyz  # B * BB * 3
        bboxes_labels_gt = semantic_labels  # B * BB
        bboxes_heading_labels_gt = heading_labels
        bboxes_heading_residuals_gt = heading_residuals
        bboxes_size_labels_gt = size_labels
        bboxes_size_residuals_gt = size_residuals
        dist_mat = tf.norm(tf.expand_dims(proposals_xyz, 2) - tf.expand_dims(bboxes_xyz_gt, 1), axis=-1)  # B * PR * BB
        bboxes_assignment = tf.argmin(dist_mat, axis=-1)  # B * PR
        min_dist = tf.reduce_min(dist_mat, axis=-1)

        positive_idxes = tf.where(min_dist < config.POSITIVE_THRES)  # Np * 2
        # with tf.control_dependencies([tf.print(tf.shape(positive_idxes))]):
        negative_idxes = tf.where(min_dist > config.NEGATIVE_THRES)  # Nn * 2
        positive_gt_idxes = tf.stack([positive_idxes[:, 0], tf.gather_nd(bboxes_assignment, positive_idxes)], axis=1)

        # objectiveness loss
        pos_obj_cls_score = tf.gather_nd(obj_cls_score, positive_idxes)
        pos_obj_cls_gt = tf.ones([tf.shape(positive_idxes)[0]], dtype=tf.int32)
        neg_obj_cls_score = tf.gather_nd(obj_cls_score, negative_idxes)
        neg_obj_cls_gt = tf.zeros([tf.shape(negative_idxes)[0]], dtype=tf.int32)
        obj_cls_loss = tf.identity(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pos_obj_cls_score, labels=pos_obj_cls_gt))
                                   + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=neg_obj_cls_score, labels=neg_obj_cls_gt)), name='obj_cls_loss')
        obj_correct = tf.concat([tf.cast(tf.nn.in_top_k(pos_obj_cls_score, pos_obj_cls_gt, 1), tf.float32),
                                 tf.cast(tf.nn.in_top_k(neg_obj_cls_score, neg_obj_cls_gt, 1), tf.float32)], axis=0, name='obj_correct')
        obj_accuracy = tf.reduce_mean(obj_correct, name='obj_accuracy')

        # center regression losses
        center_gt = tf.gather_nd(bboxes_xyz_gt, positive_gt_idxes)
        delta_predicted = tf.gather_nd(proposals_output[..., 2:5], positive_idxes)
        delta_gt = center_gt - tf.gather_nd(proposals_xyz, positive_idxes)
        center_loss = tf.reduce_mean(tf.reduce_sum(tf.losses.huber_loss(labels=delta_gt, predictions=delta_predicted, reduction=tf.losses.Reduction.NONE), axis=-1))

        # Appendix A1: chamfer loss, assignment at least one bbox to each gt bbox
        bboxes_assignment_dual = tf.argmin(dist_mat, axis=1)  # B * BB
        batch_idx = tf.tile(tf.expand_dims(tf.range(tf.shape(bboxes_assignment_dual, out_type=tf.int64)[0]), axis=-1), [1, tf.shape(bboxes_assignment_dual)[1]])  # B * BB
        delta_gt_dual = bboxes_xyz_gt - tf.gather_nd(proposals_xyz, tf.stack([batch_idx, bboxes_assignment_dual], axis=-1))  # B * BB * 3
        delta_predicted_dual = tf.gather_nd(proposals_output[..., 2:5], tf.stack([batch_idx, bboxes_assignment_dual], axis=-1))  # B * BB * 3
        center_loss_dual = tf.reduce_mean(tf.reduce_sum(tf.losses.huber_loss(labels=delta_gt_dual, predictions=delta_predicted_dual, reduction=tf.losses.Reduction.NONE), axis=-1))

        # add up
        center_loss += center_loss_dual

        # Heading loss
        heading_cls_gt = tf.gather_nd(bboxes_heading_labels_gt, positive_gt_idxes)
        heading_cls_score = tf.gather_nd(proposals_output[..., 5:5+config.NH], positive_idxes)
        heading_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=heading_cls_score, labels=heading_cls_gt))

        heading_cls_gt_onehot = tf.one_hot(heading_cls_gt,  depth=config.NH, on_value=1, off_value=0, axis=-1)  # Np * NH
        heading_residual_gt = tf.gather_nd(bboxes_heading_residuals_gt, positive_gt_idxes)  # Np
        heading_residual_predicted = tf.gather_nd(proposals_output[..., 5 + config.NH:5+2 * config.NH], positive_idxes)  # Np * NH
        heading_residual_loss = tf.losses.huber_loss(labels=heading_residual_gt,
                                                     predictions=tf.reduce_sum(heading_residual_predicted * tf.to_float(heading_cls_gt_onehot), axis=1), reduction=tf.losses.Reduction.MEAN)

        # Size loss
        size_cls_gt = tf.gather_nd(bboxes_size_labels_gt, positive_gt_idxes)
        size_cls_score = tf.gather_nd(proposals_output[..., 5+2 * config.NH:5+2 * config.NH + config.NS], positive_idxes)
        size_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=size_cls_score, labels=size_cls_gt))

        size_cls_gt_onehot = tf.one_hot(size_cls_gt, depth=config.NS, on_value=1, off_value=0, axis=-1)  # Np * NS
        size_cls_gt_onehot = tf.tile(tf.expand_dims(tf.to_float(size_cls_gt_onehot), -1), [1, 1, 3])  # Np * NS * 3
        size_residual_gt = tf.gather_nd(bboxes_size_residuals_gt, positive_gt_idxes)  # Np * 3
        size_residual_predicted = tf.reshape(tf.gather_nd(proposals_output[..., 5+2 * config.NH + config.NS:5+2 * config.NH + 4 * config.NS], positive_idxes), (-1, config.NS, 3))  # Np * NS * 3
        size_residual_loss = tf.reduce_mean(tf.reduce_sum(tf.losses.huber_loss(labels=size_residual_gt,
                                                                               predictions=tf.reduce_sum(size_residual_predicted * tf.to_float(size_cls_gt_onehot), axis=1), reduction=tf.losses.Reduction.NONE), axis=-1))

        box_loss = center_loss + 0.1 * heading_cls_loss + heading_residual_loss + 0.1 * size_cls_loss + size_residual_loss

        # semantic loss
        sem_cls_score = tf.gather_nd(proposals_output[..., -config.NC:], positive_idxes)
        sem_cls_gt = tf.gather_nd(bboxes_labels_gt, positive_gt_idxes)  # Np
        sem_cls_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sem_cls_score, labels=sem_cls_gt),
            name='sem_cls_loss')
        sem_correct = tf.cast(tf.nn.in_top_k(sem_cls_score, sem_cls_gt, 1), tf.float32, name='sem_correct')
        sem_accuracy = tf.reduce_mean(sem_correct, name='sem_accuracy')

        # This will monitor training error & accuracy (in a moving average fashion). The value will be automatically
        # 1. written to tensosrboard
        # 2. written to stat.json
        # 3. printed after each epoch
        summary.add_moving_summary(obj_accuracy, sem_accuracy)

        # Use a regex to find parameters to apply weight decay.
        # Here we apply a weight decay on all W (weight matrix) of all fc layers
        # If you don't like regex, you can certainly define the cost in any other methods.
        # no weight decay
        # wd_cost = tf.multiply(1e-5,
        #                       regularize_cost('.*/W', tf.nn.l2_loss),
        #                       name='regularize_loss')
        total_cost = vote_reg_loss + 0.5 * obj_cls_loss + 1. * box_loss + 0.1 * sem_cls_loss
        total_cost = tf.identity(total_cost, name='total_cost')
        summary.add_moving_summary(total_cost)

        # monitor histogram of all weight (of conv and fc layers) in tensorboard
        summary.add_param_summary(('.*/W', ['histogram', 'rms']))
        # the function should return the total cost to be optimized
        return total_cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-3, trainable=False)
        # This will also put the summary in tensorboard, stat.json and print in terminal,
        # but this time without moving average
        tf.summary.scalar('lr', lr)
        # opt = tf.train.MomentumOptimizer(lr, 0.9)
        opt = tf.train.AdamOptimizer(lr)

        return optimizer.apply_grad_processors(
            opt, [gradproc.MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.5)),
                  gradproc.SummaryGradient()])


if __name__=='__main__':
   pass