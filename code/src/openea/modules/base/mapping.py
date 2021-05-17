import numpy as np
import tensorflow as tf
from openea.modules.base.initializers import orthogonal_init
from openea.modules.base.initializers import bias_init
from openea.modules.base.losses import mapping_loss
from openea.modules.base.losses import calibration_loss
from openea.modules.base.losses import mapping_concat_loss
from openea.modules.base.losses import calibration_concat_loss
from openea.modules.base.losses import onto_align_loss
from openea.modules.base.losses import nolinear_mapping_loss
from openea.modules.base.optimizers import generate_optimizer


def add_mapping_module(model):
    with tf.name_scope('seed_links_placeholder'):
        model.seed_entities1 = tf.placeholder(tf.int32, shape=[None])
        model.seed_entities2 = tf.placeholder(tf.int32, shape=[None])
    with tf.name_scope('seed_links_lookup'):
        tes1 = tf.nn.embedding_lookup(model.ent_embeds, model.seed_entities1)
        tes2 = tf.nn.embedding_lookup(model.ent_embeds, model.seed_entities2)
    with tf.name_scope('mapping_loss'):
        model.mapping_loss = model.args.alpha * mapping_loss(tes1, tes2, model.mapping_mat, model.eye_mat)
        model.mapping_optimizer = generate_optimizer(model.mapping_loss, model.args.learning_rate,
                                                     opt=model.args.optimizer)


def add_ent_onto_mapping_module(model, fusion_way=None):
    with tf.name_scope('seed_links_placeholder'):
        model.seed_entities1 = tf.placeholder(tf.int32, shape=[None])
        model.seed_entities2 = tf.placeholder(tf.int32, shape=[None])
        model.seed_ontologies1 = tf.placeholder(tf.int32, shape=[None])
        model.seed_ontologies2 = tf.placeholder(tf.int32, shape=[None])
    with tf.name_scope('seed_links_lookup'):
        ent1 = tf.nn.embedding_lookup(model.ent_embeds, model.seed_entities1)
        ent2 = tf.nn.embedding_lookup(model.ent_embeds, model.seed_entities2)
        onto1 = tf.nn.embedding_lookup(model.onto_ent_embeds, model.seed_ontologies1)
        onto2 = tf.nn.embedding_lookup(model.onto_ent_embeds, model.seed_ontologies2)
    with tf.name_scope('mapping_loss'):
        if fusion_way is None:
            model.mapping_loss = model.args.alpha * mapping_loss(ent1, ent2, model.mapping_mat, model.eye_mat)
        elif fusion_way == 'add':
            ent_align_loss = model.args.alpha * mapping_loss(ent1, ent2, model.mapping_mat, model.eye_mat)
            onto_align_loss = model.args.gamma * calibration_loss(onto1, onto2)
            model.mapping_loss = ent_align_loss + onto_align_loss
        elif fusion_way == 'mapping_concat':
            model.mapping_loss = model.args.alpha * mapping_concat_loss(ent1, ent2, onto1, onto2, model.mapping_mat,
                                                                        onto_weight=model.args.gamma, eye=model.eye_mat)
        elif fusion_way == 'calibration_concat':
            model.mapping_loss = model.args.alpha * calibration_concat_loss(ent1, ent2, onto1, onto2, model.args.gamma)
        else:
            raise Exception("unvaild fusion way .")
        model.mapping_optimizer = generate_optimizer(model.mapping_loss, model.args.learning_rate,
                                                     opt=model.args.optimizer)


def add_mapping_module_with_ontoAlign(model):
    # 除了实体对齐之外，还需要它们对齐的本体对齐
    with tf.name_scope('seed_links_placeholder'):
        model.seed_entities1 = tf.placeholder(tf.int32, shape=[None])
        model.seed_entities2 = tf.placeholder(tf.int32, shape=[None])
    with tf.name_scope('seed_links_lookup'):
        tes1 = tf.nn.embedding_lookup(model.ent_embeds, model.seed_entities1)
        tes2 = tf.nn.embedding_lookup(model.ent_embeds, model.seed_entities2)
    with tf.name_scope('mapping_loss'):
        insalign_loss = model.args.alpha * mapping_loss(tes1, tes2, model.mapping_mat, model.eye_mat)
        if model.args.onto_mapping_method == 'linear':
            ontoalign_loss = model.args.beta * onto_align_loss(tes1, tes2, model.onto_mapping_mat,
                                                               eye=model.onto_eye_mat, method='linear')
        elif model.args.onto_mapping_method == 'nolinear':
            ontoalign_loss = model.args.beta * onto_align_loss(tes1, tes2, model.onto_mapping_mat,
                                                               biases=model.onto_mapping_bias, method='nolinear')
        model.mapping_loss = insalign_loss + ontoalign_loss
        model.mapping_optimizer = generate_optimizer(model.mapping_loss, model.args.learning_rate,
                                                     opt=model.args.optimizer)


def add_onto_mapping_module(model, orthogonal=True, method='linear'):
    # 实体和其对应的本体对齐
    with tf.name_scope('cv_links_placeholder'):
        model.entities = tf.placeholder(tf.int32, shape=[None])
        model.ontologies = tf.placeholder(tf.int32, shape=[None])
    with tf.name_scope('cv_links_lookup'):
        tes1 = tf.nn.embedding_lookup(model.ent_embeds, model.entities)
        tes2 = tf.nn.embedding_lookup(model.onto_ent_embeds, model.ontologies)
    with tf.name_scope('onto_mapping_loss'):
        if method == 'linear':
            model.onto_mapping_loss = model.args.gamma * mapping_loss(tes1, tes2, model.onto_mapping_mat,
                                                                      model.onto_eye_mat, orthogonal=orthogonal)
        elif method == 'nolinear':
            model.onto_mapping_loss = model.args.gamma * nolinear_mapping_loss(tes1, tes2, model.onto_mapping_mat,
                                                                               model.onto_mapping_bias, activation='tanh')
        model.onto_mapping_optimizer = generate_optimizer(model.onto_mapping_loss, model.args.learning_rate,
                                                          opt=model.args.optimizer)


def add_mapping_variables(model):
    with tf.variable_scope('kgs' + 'mapping'):
        model.mapping_mat = orthogonal_init([model.args.dim, model.args.dim], 'mapping_matrix')
        model.eye_mat = tf.constant(np.eye(model.args.dim), dtype=tf.float32, name='eye')


def add_mlp_variables(model, name, shape):
    with tf.variable_scope(name):
        model.mlp_weights = orthogonal_init(shape, 'mlp_weights')
        model.mlp_bias = bias_init([shape[-1], ], 'mlp_bias')


def add_onto_mapping_variables(model, method='linear'):
    with tf.variable_scope('onto' + 'mapping'):
        if method == 'linear':
            model.onto_mapping_mat = orthogonal_init([model.args.dim, model.args.onto_dim], 'onto_mapping_matrix')
            model.onto_eye_mat = tf.constant(np.eye(model.args.onto_dim), dtype=tf.float32, name='onto_eye')
        elif method == 'nolinear':
            model.onto_mapping_mat = orthogonal_init([model.args.dim, model.args.onto_dim], 'onto_mapping_matrix')
            model.onto_mapping_bias = bias_init([model.args.onto_dim, ], 'onto_mapping_biases')
