import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
from tensorpack import *
import numpy as np
from utils import pointnet_sa_module, pointnet_fp_module
import config


class Model(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, [None, config.POINT_NUM , 3], 'points'),
                tf.placeholder(tf.float32, [None, None, 3], 'bboxes'),
                tf.placeholder(tf.int32, (None, config.PROPOSAL_NUM), 'objectiveness'),
                tf.placeholder(tf.int32, (None, config.PROPOSAL_NUM), 'semantic_labels'),
                ]

    def build_graph(self, x, bboxes_gt, obj_gt, sem_gt):
        l0_xyz = tf.slice(x, [0, 0, 0], [-1, -1, 3])
        l0_points = tf.slice(x, [0, 0, 3], [-1, -1, 3])

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
        offset = tf.concat([seeds_xyz, seeds_points], 2)
        units = [256, 256, 256 + 3]
        for i in range(len(units)):
            offset = FullyConnected('voting%d' % i, offset, units[i], activation=BNReLU if i < len(units) - 1 else None)

        # B * N * 3
        vote_reg_loss = tf.reduce_mean(tf.norm(offset[:, :, :3] - offset_gt, axis=-1), name='vote_reg_loss')
        votes = tf.concat([seeds_xyz, seeds_points]) + offset
        votes_xyz = votes[:, :, :3]
        votes_points = votes[:, :, 3:]

        # Proposal Module layers
        # Farthest point sampling on seeds
        proposals_xyz, proposals_output, _ = pointnet_sa_module(seeds_xyz, votes_points, npoint=config.PROPOSAL_NUM,
                                                                radius=0.3, nsample=64, mlp=[128, 128, 128],
                                                                mlp2=[128, 128, 5+2 * config.NH+4 * config.NS+config.NC],
                                                                group_all=False, scope='proposal',
                                                                group_xyz=votes_xyz)

        # calculate positive and negative proposal idxes
        bboxes_xyz_gt = bboxes_gt[:, :, :3]
        dist_mat = tf.norm(tf.expand_dims(proposals_xyz, 1) - tf.expand_dims(bboxes_xyz_gt, 2), axis=-1)  # B * PR * BB
        min_dist = tf.reduce_min(dist_mat, axis=-1)

        positive_idxes = tf.where(min_dist < config.POSITIVE_THRES)  # N * 2
        negative_idxes = tf.where(min_dist > config.NEGATIVE_THRES)  # N * 2

        # objectiveness loss
        obj_cls_score = proposals_output[..., :2]
        obj_cls_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=obj_cls_score, labels=obj_gt), name='obj_cls_loss')

        sem_cls_score = proposals_output[..., -config.NC:]



        sem_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sem_cls_score, labels=sem_gt), name='sem_cls_loss')

        obj_correct = tf.cast(tf.nn.in_top_k(obj_cls_score, obj_gt, 1), tf.float32, name='obj_correct')
        obj_accuracy = tf.reduce_mean(obj_correct, name='obj_accuracy')

        sem_correct = tf.cast(tf.nn.in_top_k(sem_cls_score, sem_gt, 1), tf.float32, name='sem_correct')
        sem_accuracy = tf.reduce_mean(sem_correct, name='sem_accuracy')

        # Center regression losses
        center_dist = tf.norm(center_label - end_points['center'], axis=-1)
        center_loss = huber_loss(center_dist, delta=2.0)
        tf.summary.scalar('center loss', center_loss)
        stage1_center_dist = tf.norm(center_label - \
                                     end_points['stage1_center'], axis=-1)
        stage1_center_loss = huber_loss(stage1_center_dist, delta=1.0)
        tf.summary.scalar('stage1 center loss', stage1_center_loss)

        # Heading loss
        heading_class_loss = tf.reduce_mean( \
            tf.nn.sparse_softmax_cross_entropy_with_logits( \
                logits=end_points['heading_scores'], labels=heading_class_label))
        tf.summary.scalar('heading class loss', heading_class_loss)

        hcls_onehot = tf.one_hot(heading_class_label,
                                 depth=NUM_HEADING_BIN,
                                 on_value=1, off_value=0, axis=-1)  # BxNUM_HEADING_BIN
        heading_residual_normalized_label = \
            heading_residual_label / (np.pi / NUM_HEADING_BIN)
        heading_residual_normalized_loss = huber_loss(tf.reduce_sum( \
            end_points['heading_residuals_normalized'] * tf.to_float(hcls_onehot), axis=1) - \
                                                      heading_residual_normalized_label, delta=1.0)
        tf.summary.scalar('heading residual normalized loss',
                          heading_residual_normalized_loss)

        # Size loss
        size_class_loss = tf.reduce_mean( \
            tf.nn.sparse_softmax_cross_entropy_with_logits( \
                logits=end_points['size_scores'], labels=size_class_label))
        tf.summary.scalar('size class loss', size_class_loss)

        scls_onehot = tf.one_hot(size_class_label,
                                 depth=NUM_SIZE_CLUSTER,
                                 on_value=1, off_value=0, axis=-1)  # BxNUM_SIZE_CLUSTER
        scls_onehot_tiled = tf.tile(tf.expand_dims( \
            tf.to_float(scls_onehot), -1), [1, 1, 3])  # BxNUM_SIZE_CLUSTERx3
        predicted_size_residual_normalized = tf.reduce_sum( \
            end_points['size_residuals_normalized'] * scls_onehot_tiled, axis=[1])  # Bx3

        mean_size_arr_expand = tf.expand_dims( \
            tf.constant(g_mean_size_arr, dtype=tf.float32), 0)  # 1xNUM_SIZE_CLUSTERx3
        mean_size_label = tf.reduce_sum( \
            scls_onehot_tiled * mean_size_arr_expand, axis=[1])  # Bx3
        size_residual_label_normalized = size_residual_label / mean_size_label
        size_normalized_dist = tf.norm( \
            size_residual_label_normalized - predicted_size_residual_normalized,
            axis=-1)
        size_residual_normalized_loss = huber_loss(size_normalized_dist, delta=1.0)
        tf.summary.scalar('size residual normalized loss',
                          size_residual_normalized_loss)

        # This will monitor training error & accuracy (in a moving average fashion). The value will be automatically
        # 1. written to tensosrboard
        # 2. written to stat.json
        # 3. printed after each epoch
        summary.add_moving_summary(obj_accuracy, sem_accuracy)

        # Use a regex to find parameters to apply weight decay.
        # Here we apply a weight decay on all W (weight matrix) of all fc layers
        # If you don't like regex, you can certainly define the cost in any other methods.
        wd_cost = tf.multiply(1e-5,
                              regularize_cost('.*/W', tf.nn.l2_loss),
                              name='regularize_loss')
        total_cost = tf.add_n([cost, wd_cost], name='total_cost')
        summary.add_moving_summary(cost, wd_cost, total_cost)

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
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,6))
        net, _ = get_model(inputs, tf.constant(True))
        print(net)