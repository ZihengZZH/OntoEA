import numpy as np
import tensorflow as tf
from openea.modules.base.initializers import *


def get_loss_func(phs, prs, pts, nhs, nrs, nts, args):
    triple_loss = None
    if args.loss == 'margin-based':
        triple_loss = margin_loss(phs, prs, pts, nhs, nrs, nts, args.margin, args.loss_norm)
    elif args.loss == 'logistic':
        triple_loss = logistic_loss(phs, prs, pts, nhs, nrs, nts, args.loss_norm)
    elif args.loss == 'limited':
        triple_loss = limited_loss(phs, prs, pts, nhs, nrs, nts, args.pos_margin, args.neg_margin, args.loss_norm)
    return triple_loss


def margin_loss(phs, prs, pts, nhs, nrs, nts, margin, loss_norm):
    with tf.name_scope('margin_loss_distance'):
        pos_distance = phs + prs - pts
        neg_distance = nhs + nrs - nts
    with tf.name_scope('margin_loss'):
        if loss_norm == 'L1':  # L1 normal
            pos_score = tf.reduce_sum(tf.abs(pos_distance), axis=1)
            neg_score = tf.reduce_sum(tf.abs(neg_distance), axis=1)
        else:  # L2 normal
            pos_score = tf.reduce_sum(tf.square(pos_distance), axis=1)
            neg_score = tf.reduce_sum(tf.square(neg_distance), axis=1)
        loss = tf.reduce_sum(tf.nn.relu(tf.constant(margin) + pos_score - neg_score), name='margin_loss')
    return loss


def positive_loss(phs, prs, pts, loss_norm):
    with tf.name_scope('positive_loss_distance'):
        pos_distance = phs + prs - pts
    with tf.name_scope('positive_loss_score'):
        if loss_norm == 'L1':  # L1 score
            pos_score = tf.reduce_sum(tf.abs(pos_distance), axis=1)
        else:  # L2 score
            pos_score = tf.reduce_sum(tf.square(pos_distance), axis=1)
        loss = tf.reduce_sum(pos_score, name='positive_loss')
    return loss


def hierarchical_loss(phs, pts, mat, bias, loss_norm, activation='tanh'):
    # 类似JOIE方式
    with tf.name_scope('subclassof_loss_distance'):
        if activation == 'tanh':
            mapping = tf.matmul(phs, mat) + bias
            mapping = tf.nn.tanh(mapping)
        elif activation == 'linear':
            mapping = phs
        pos_distance = pts - mapping
    with tf.name_scope('subclassof_loss_score'):
        if loss_norm == 'L1':  # L1 score
            pos_score = tf.reduce_sum(tf.abs(pos_distance), axis=1)
        else:  # L2 score
            pos_score = tf.reduce_sum(tf.square(pos_distance), axis=1)
        loss = tf.reduce_sum(pos_score, name='hierarchical_loss')
    return loss


def type_mat_loss(onto_emb, onto_mat, loss_norm):
    with tf.name_scope('emb_cos_similarity'):
        norm = tf.sqrt(tf.reduce_sum(onto_emb * onto_emb, 1))
        type_sim = tf.matmul(onto_emb, tf.transpose(onto_emb))
        type_sim = tf.div(type_sim, norm * norm + 1e-8)
        print('type sim shape: ', type_sim.shape)
        print('onto mat shape: ', onto_mat.shape)
        distance = type_sim - onto_mat
    with tf.name_scope('type_mat_loss'):
        if loss_norm == 'L1':
            score = tf.reduce_sum(tf.abs(distance), axis=1)
        else:  # L2 score
            score = tf.reduce_sum(tf.square(distance), axis=1)
        loss = tf.reduce_sum(score, name='type_mat_loss')
    return loss


def limited_loss(phs, prs, pts, nhs, nrs, nts, pos_margin, neg_margin, loss_norm, balance=1.0):
    with tf.name_scope('limited_loss_distance'):
        pos_distance = phs + prs - pts
        neg_distance = nhs + nrs - nts
    with tf.name_scope('limited_loss_score'):
        if loss_norm == 'L1':  # L1 score
            pos_score = tf.reduce_sum(tf.abs(pos_distance), axis=1)
            neg_score = tf.reduce_sum(tf.abs(neg_distance), axis=1)
        else:  # L2 score
            pos_score = tf.reduce_sum(tf.square(pos_distance), axis=1)
            neg_score = tf.reduce_sum(tf.square(neg_distance), axis=1)
        pos_loss = tf.reduce_sum(tf.nn.relu(pos_score - tf.constant(pos_margin)))
        neg_loss = tf.reduce_sum(tf.nn.relu(tf.constant(neg_margin) - neg_score))
        loss = tf.add(pos_loss, balance * neg_loss, name='limited_loss')
    return loss


def hier_limited_loss(phs, prs, pts, nhs, nrs, nts, pos_margin, neg_margin, loss_norm, balance=1.0,
                      weights=None, bias=None):
    with tf.name_scope('hier_limited_loss_distance'):
        if weights is not None:
            pos_distance = tf.nn.l2_normalize(tf.tanh(tf.matmul(phs, weights) + bias), 1) - pts  # 归一化向量
            neg_distance = tf.nn.l2_normalize(tf.tanh(tf.matmul(nhs, weights) + bias), 1) - nts  # 归一化向量
        else:
            pos_distance = phs - pts
            neg_distance = nhs - nts
    with tf.name_scope('hier_limited_loss_score'):
        if loss_norm == 'L1':  # L1 score
            pos_score = tf.reduce_sum(tf.abs(pos_distance), axis=1)
            neg_score = tf.reduce_sum(tf.abs(neg_distance), axis=1)
        else:  # L2 score
            pos_score = tf.reduce_sum(tf.square(pos_distance), axis=1)
            neg_score = tf.reduce_sum(tf.square(neg_distance), axis=1)
        pos_loss = tf.reduce_sum(tf.nn.relu(pos_score - tf.constant(pos_margin)))
        neg_loss = tf.reduce_sum(tf.nn.relu(tf.constant(neg_margin) - neg_score))
        loss = tf.add(pos_loss, balance * neg_loss, name='hier_limited_loss')
    return loss


def ent2onto_limited_loss(phs, pts, nhs, nts, pos_margin, neg_margin, loss_norm, balance=1.0,
                          weights=None, bias=None):
    with tf.name_scope('ent2onto_limited_loss_distance'):
        if weights is not None:
            pos_distance = tf.nn.l2_normalize(tf.nn.tanh(tf.matmul(phs, weights) + bias), 1) - pts
            neg_distance = tf.nn.l2_normalize(tf.nn.tanh(tf.matmul(nhs, weights) + bias), 1) - nts
        else:
            pos_distance = phs - pts
            neg_distance = nhs - nts
    with tf.name_scope('ent2onto_limited_loss_score'):
        if loss_norm == 'L1':  # L1 score
            pos_score = tf.reduce_sum(tf.abs(pos_distance), axis=1)
            neg_score = tf.reduce_sum(tf.abs(neg_distance), axis=1)
        else:  # L2 score
            pos_score = tf.reduce_sum(tf.square(pos_distance), axis=1)
            neg_score = tf.reduce_sum(tf.square(neg_distance), axis=1)
        pos_loss = tf.reduce_sum(tf.nn.relu(pos_score - tf.constant(pos_margin)))
        neg_loss = tf.reduce_sum(tf.nn.relu(tf.constant(neg_margin) - neg_score))
        loss = tf.add(pos_loss, balance * neg_loss, name='ent2onto_limited_loss')
    return loss


def logistic_loss(phs, prs, pts, nhs, nrs, nts, loss_norm):
    with tf.name_scope('logistic_loss_distance'):
        pos_distance = phs + prs - pts
        neg_distance = nhs + nrs - nts
    with tf.name_scope('logistic_loss_score'):
        if loss_norm == 'L1':  # L1 score
            pos_score = tf.reduce_sum(tf.abs(pos_distance), axis=1)
            neg_score = tf.reduce_sum(tf.abs(neg_distance), axis=1)
        else:  # L2 score
            pos_score = tf.reduce_sum(tf.square(pos_distance), axis=1)
            neg_score = tf.reduce_sum(tf.square(neg_distance), axis=1)
        pos_loss = tf.reduce_sum(tf.log(1 + tf.exp(pos_score)))
        neg_loss = tf.reduce_sum(tf.log(1 + tf.exp(-neg_score)))
        loss = tf.add(pos_loss, neg_loss, name='logistic_loss')
    return loss


def mapping_loss(tes1, tes2, mapping, eye=None, orthogonal=True):
    mapped_tes2 = tf.matmul(tes1, mapping)   # M*k1 = k2
    map_loss = tf.reduce_sum(tf.reduce_sum(tf.pow(tes2 - mapped_tes2, 2), 1))
    orthogonal_loss = 0
    if orthogonal:
        if eye is None:
            eye = tf.constant(np.eye(mapping.shape[-1]), dtype=tf.float32, name='eye')
        # 映射矩阵正交约束
        orthogonal_loss = tf.reduce_sum(tf.reduce_sum(tf.pow(tf.matmul(mapping, mapping, transpose_b=True) - eye, 2), 1))
    return map_loss + orthogonal_loss


def mapping_limit_loss(tes1, tes2, neg_tes1, neg_tes2, mapping, pos_margin, neg_margin, balance=1.0,
                       eye=None, orthogonal=True):
    mapped_tes2 = tf.matmul(tes1, mapping)   # M*k1 = k2
    mapped_neg_tes2 = tf.matmul(neg_tes1, mapping)
    pos_score = tf.reduce_sum(tf.reduce_sum(tf.pow(tes2 - mapped_tes2, 2), 1))
    neg_score = tf.reduce_sum(tf.reduce_sum(tf.pow(neg_tes2 - mapped_neg_tes2, 2), 1))
    pos_loss = tf.reduce_sum(tf.nn.relu(pos_score - tf.constant(pos_margin)))
    neg_loss = tf.reduce_sum(tf.nn.relu(tf.constant(neg_margin) - neg_score))
    loss = tf.add(pos_loss, balance * neg_loss, name='mapping_limited_loss')
    orthogonal_loss = 0
    if orthogonal:
        if eye is None:
            eye = tf.constant(np.eye(mapping.shape[-1]), dtype=tf.float32, name='eye')
        # 映射矩阵正交约束
        orthogonal_loss = tf.reduce_sum(tf.reduce_sum(tf.pow(tf.matmul(mapping, mapping, transpose_b=True) - eye, 2), 1))
    return loss + orthogonal_loss


def mapping_concat_loss(ent1, ent2, onto1, onto2, mapping, onto_weight=1.0, eye=None, orthogonal=True):
    mapped_ent2 = tf.matmul(ent1, mapping)   # M*k1 = k2
    tes1 = tf.concat([mapped_ent2, onto_weight*onto1], axis=1)
    tes2 = tf.concat([ent2, onto_weight*onto2], axis=1)
    map_loss = tf.reduce_sum(tf.reduce_sum(tf.pow(tes1 - tes2, 2), 1))
    orthogonal_loss = 0
    if orthogonal:
        if eye is None:
            eye = tf.constant(np.eye(mapping.shape[-1]), dtype=tf.float32, name='eye')
        # 映射矩阵正交约束
        orthogonal_loss = tf.reduce_sum(tf.reduce_sum(tf.pow(tf.matmul(mapping, mapping, transpose_b=True) - eye, 2), 1))
    return map_loss + orthogonal_loss


def calibration_concat_loss(ent1, ent2, onto1, onto2, onto_weight=1.0):
    tes1 = tf.concat([ent1, onto_weight*onto1], axis=1)
    tes2 = tf.concat([ent2, onto_weight*onto2], axis=1)
    loss = tf.reduce_sum(tf.reduce_sum(tf.pow(tes1 - tes2, 2), 1))
    return loss


def calibration_loss(tes1, tes2):
    loss = tf.reduce_sum(tf.reduce_sum(tf.pow(tes2 - tes1, 2), 1))
    return loss


def fusion_ent_onto_loss(ent1, ent2, onto1, onto2, mode='calibration',
                         weights=None, bias=None, activation='linear'):
    '''
    实体和本体融合的Loss
    '''
    msg1 = tf.concat([ent1, onto1], axis=-1)
    msg2 = tf.concat([ent2, onto2], axis=-1)
    print('msg1 shape : ', msg1.shape)
    if mode == 'ent_calibration':
        return calibration_loss(ent1, ent2)
    elif mode == 'concat_calibration':
        return calibration_loss(msg1, msg2)
    elif mode == 'add_calibration':
        return calibration_loss(ent1 + onto1, ent2 + onto2)
        # return calibration_loss(ent1,ent2) + calibration_loss(onto1,onto2)
    elif mode == 'add_mapping':
        return mapping_loss(ent1+onto1, ent2+onto2, mapping=weights)
        # return mapping_loss(ent1,ent2,mapping=weights) + calibration_loss(onto1,onto2)
    elif mode == 'ent_mapping':                 # MtranE原始方式
        return mapping_loss(ent1, ent2, mapping=weights)
    elif mode == 'concat_mapping':
        if activation == 'tanh':
            msg1 = tf.nn.tanh(tf.matmul(msg1, weights) + bias)
            msg2 = tf.nn.tanh(tf.matmul(msg2, weights) + bias)
        elif activation == 'linear':
            msg1 = tf.matmul(msg1, weights) + bias
            msg2 = tf.matmul(msg2, weights) + bias
        else:
            raise Exception("unvaild activation.")
        print('msg1 shape : ', msg1.shape)
        return calibration_loss(msg1, msg2)


def onto_align_loss(tes1, tes2, mapping, eye=None, biases=None, method='linear', activation='tanh'):
    mapped_tes1 = tf.matmul(tes1, mapping)  # 实体对应的本体映射
    mapped_tes2 = tf.matmul(tes2, mapping)  # 实体对应的本体映射
    map_loss = None
    if method == 'linear':
        map_loss = tf.reduce_sum(tf.reduce_sum(tf.pow(mapped_tes1 - mapped_tes2, 2), 1))
    elif method == 'nolinear':
        mapped_tes1 = tf.nn.tanh(mapped_tes1 + biases)
        mapped_tes2 = tf.nn.tanh(mapped_tes2 + biases)
        map_loss = tf.reduce_sum(tf.reduce_sum(tf.pow(mapped_tes1 - mapped_tes2, 2), 1))
    # 映射矩阵正交约束
    # orthogonal_loss = tf.reduce_sum(tf.reduce_sum(tf.pow(tf.matmul(mapping, mapping, transpose_b=True) - eye, 2), 1))
    return map_loss


def nolinear_mapping_loss(tes1, tes2, mapping, bias, activation='tanh'):
    mapped_tes2 = tf.matmul(tes1, mapping) + bias  # M*k1 = k2
    if activation == 'tanh':
        mapped_tes2 = tf.nn.tanh(mapped_tes2)
    elif activation == 'relu':
        mapped_tes2 = tf.nn.relu(mapped_tes2)
    elif activation == 'sigmoid':
        mapped_tes2 = tf.nn.sigmoid(mapped_tes2)
    map_loss = tf.reduce_sum(tf.reduce_sum(tf.pow(tes2 - mapped_tes2, 2), 1))
    return map_loss
