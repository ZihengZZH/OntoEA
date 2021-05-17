import gc
import math
import random
import time
import numpy as np
import tensorflow as tf
import multiprocessing as mp
import openea.modules.load.read as rd
import openea.modules.train.batch as bat
from openea.modules.finding.evaluation import early_stop
from openea.modules.finding.evaluation import valid
from openea.modules.finding.evaluation import test
from openea.modules.utils.util import task_divide
from openea.modules.base.optimizers import generate_optimizer
from openea.modules.utils.util import load_session
from openea.modules.base.initializers import init_embeddings
from openea.modules.base.initializers import init_mlp
from openea.modules.base.losses import positive_loss
from openea.modules.base.losses import hier_limited_loss
from openea.modules.base.losses import limited_loss
from openea.modules.base.losses import ent2onto_limited_loss
from openea.models.basic_model import BasicModel
from openea.modules.train import sample
from openea.modules.finding.similarity import sim
from openea.modules.base.losses import mapping_limit_loss
from openea.modules.base.losses import calibration_loss


class OntoEA(BasicModel):
    def __init__(self):
        super().__init__()
        self.cvlink_weights = None
        self.cvlink_bias = None
        self.word_embed = None
        self.use_word_embed_init = None
        self.use_word_embed_init_onto = None
        self.word_embed_init_zh = None
        self.use_alter_label = None

    def init(self):
        self.word_embed = self.args.word_embed
        self.use_word_embed_init = self.args.use_word_embed_init
        self.use_word_embed_init_onto = self.args.use_word_embed_init_onto
        self.word_embed_init_zh = self.args.word_embed_init_zh
        self.use_alter_label = self.args.use_alter_label
        self._define_variables()                # 定义实体embedding和本体embedding
        self._define_mapping_variables()        # 实体对齐映射参数，即Me1-e2中的M
        self._define_embed_graph()              # 实体KG嵌入表示
        self._define_onto_embed_graph()         # 本体KG嵌入表示
        self._define_ent2onto_mapping_graph()   # membership link 嵌入表示
        self._define_mapping_graph(fusion_way=self.args.fusion_way)   # 本体+实体对齐Loss

        if self.args.likelihood_vaild:
            self._define_onto_likelihood_graph()  # 冲突矩阵CCM似然loss

        self.session = load_session()
        tf.global_variables_initializer().run(session=self.session)

        # customize parameters
        assert self.args.init == 'unit'
        assert self.args.alignment_module == 'mapping'
        assert self.args.optimizer == 'Adagrad'
        assert self.args.eval_metric == 'inner'
        assert self.args.ent_l2_norm is True

        assert self.args.alpha > 1

    def _read_word2vec(self, file_path, dim):
        print('\n', file_path)
        word2vec = dict()
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip('\n').split(' ')
                if len(line) != dim + 1:
                    continue
                try:
                    v = np.array(list(map(float, line[1:])), dtype=np.float64)
                    word2vec[line[0].lower()] = v
                except:
                    continue
        file.close()
        return word2vec

    def _init_embeddings_word2vec(self, word2vec, shape, dict_kg1, dict_kg2, name, name_dict=None):
        with tf.name_scope('word2vec_init'):
            std = 1.0 / math.sqrt(shape[1])
            embeds = np.random.normal(scale=std, size=shape)
            kv_dict = {}
            for key, val in dict_kg1.items():
                if val not in kv_dict.keys():
                    kv_dict[val] = key
            for key, val in dict_kg2.items():
                if val not in kv_dict.keys():
                    kv_dict[val] = key
            for idx in range(len(embeds)):
                if self.use_alter_label == 1 and 'onto' not in name:
                    if kv_dict[idx] in name_dict.keys():
                        nns = name_dict[kv_dict[idx]].split(' ')
                    else:
                        nns = ['']
                else:
                    # industry dataset (MED-BBK-9K)
                    if '@@' in kv_dict[idx]:
                        nns = kv_dict[idx].split('@@')[-1]
                    else:
                        nns = kv_dict[idx].split('/')[-1].split('_')
                    if self.word_embed_init_zh == 1:
                        nns = list(nns)
                for nn in nns:
                    nn = nn.lower()
                    if nn in word2vec.keys():
                        embeds[idx] += word2vec[nn]
        embeddings = tf.Variable(np.matrix(embeds), name=name, dtype=tf.float32)
        return tf.nn.l2_normalize(embeddings, 1)

    ######################################################
    ################### 定义每个模块    ####################
    ######################################################

    def _define_variables(self):
        with tf.variable_scope('relational' + 'embeddings'):
            # kgs.entities_num ：两个KG的实体总数;
            # args.init : embedding初始化方式，例如'unit';
            # ent_l2_norm：是否L2标准化
            if self.use_word_embed_init == 1:
                word2vec = self._read_word2vec(self.word_embed, self.args.dim)
                self.ent_embeds = self._init_embeddings_word2vec(
                    word2vec,
                    [self.kgs.entities_num, self.args.dim],
                    self.kgs.kg1.entities_id_dict,
                    self.kgs.kg2.entities_id_dict,
                    'ent_embeds',
                    self.kgs.name_dict
                )
                self.rel_embeds = self._init_embeddings_word2vec(
                    word2vec,
                    [self.kgs.relations_num, self.args.dim],
                    self.kgs.kg1.relations_id_dict,
                    self.kgs.kg2.relations_id_dict,
                    'rel_embeds',
                    self.kgs.name_dict
                )
            else:
                self.ent_embeds = init_embeddings(
                    [self.kgs.entities_num, self.args.dim],
                    'ent_embeds',
                    self.args.init, self.args.ent_l2_norm
                )
                self.rel_embeds = init_embeddings(
                    [self.kgs.relations_num, self.args.dim],
                    'rel_embeds',
                    self.args.init, self.args.rel_l2_norm
                )
            if self.use_word_embed_init_onto == 1:
                word2vec = self._read_word2vec(self.word_embed, self.args.dim)
                self.onto_ent_embeds = self._init_embeddings_word2vec(
                    word2vec,
                    [self.kgs.onto_entities_num, self.args.onto_dim],
                    self.kgs.onto_class_id_dict,
                    self.kgs.onto_class_id_dict,
                    'onto_ent_embeds',
                    self.kgs.name_dict
                )
                self.onto_rel_embeds = self._init_embeddings_word2vec(
                    word2vec,
                    [self.kgs.onto_relations_num, self.args.onto_dim],
                    self.kgs.onto_metarel_id_dict,
                    self.kgs.onto_metarel_id_dict,
                    'onto_rel_embeds',
                    self.kgs.name_dict
                )
            else:
                self.onto_ent_embeds = init_embeddings(
                    [self.kgs.onto_entities_num, self.args.onto_dim],
                    'onto_ent_embeds',
                    self.args.init, self.args.ent_l2_norm
                )
                self.onto_rel_embeds = init_embeddings(
                    [self.kgs.onto_relations_num, self.args.onto_dim],
                    'onto_rel_embeds',
                    self.args.init, self.args.rel_l2_norm
                )

    def _define_embed_graph(self):
        with tf.name_scope('triple_placeholder'):
            self.pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.pos_rs = tf.placeholder(tf.int32, shape=[None])
            self.pos_ts = tf.placeholder(tf.int32, shape=[None])
            self.neg_hs = tf.placeholder(tf.int32, shape=[None])
            self.neg_rs = tf.placeholder(tf.int32, shape=[None])
            self.neg_ts = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('triple_lookup'):
            phs = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.ent_embeds, self.pos_hs), 1)
            prs = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.rel_embeds, self.pos_rs), 1)
            pts = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.ent_embeds, self.pos_ts), 1)
            nhs = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.ent_embeds, self.neg_hs), 1)
            nrs = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.rel_embeds, self.neg_rs), 1)
            nts = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.ent_embeds, self.neg_ts), 1)
        with tf.name_scope('triple_loss'):
            if self.args.ent_emb_loss == 'pos_loss':
                # 实体正样本损失
                self.triple_loss = positive_loss(phs, prs, pts, 'L2')
                self.triple_optimizer = generate_optimizer(self.triple_loss, self.args.learning_rate, opt=self.args.optimizer)
            elif self.args.ent_emb_loss == "limit_loss":
                self.triple_loss = limited_loss(phs, prs, pts, nhs, nrs, nts,
                                                self.args.pos_margin, self.args.neg_margin,
                                                self.args.loss_norm, balance=self.args.neg_margin_balance)
                self.triple_optimizer = generate_optimizer(self.triple_loss, self.args.learning_rate,
                                                           opt=self.args.optimizer)

    def _define_onto_embed_graph(self):
        with tf.name_scope('onto_triple_placeholder'):
            self.onto_pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.onto_pos_rs = tf.placeholder(tf.int32, shape=[None])
            self.onto_pos_ts = tf.placeholder(tf.int32, shape=[None])
            self.onto_neg_hs = tf.placeholder(tf.int32, shape=[None])
            self.onto_neg_rs = tf.placeholder(tf.int32, shape=[None])
            self.onto_neg_ts = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('onto_triple_lookup'):
            onto_phs = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.onto_ent_embeds, self.onto_pos_hs), 1)
            onto_prs = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.onto_rel_embeds, self.onto_pos_rs), 1)
            onto_pts = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.onto_ent_embeds, self.onto_pos_ts), 1)
            onto_nhs = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.onto_ent_embeds, self.onto_neg_hs), 1)
            onto_nrs = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.onto_rel_embeds, self.onto_neg_rs), 1)
            onto_nts = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.onto_ent_embeds, self.onto_neg_ts), 1)
        with tf.name_scope('onto_triple_loss'):
            if self.args.onto_training_method == 'transe':
                self.onto_triple_loss = limited_loss(onto_phs, onto_prs, onto_pts, onto_nhs, onto_nrs, onto_nts,
                                                     self.args.pos_margin, self.args.neg_margin,
                                                     self.args.loss_norm, balance=self.args.neg_margin_balance)
            elif self.args.onto_training_method == 'hier':
                self.hier_weights, self.hier_bias = init_mlp([self.args.onto_dim, self.args.onto_dim],
                                                             'onto_hier_weights', 'onto_hier_bias',
                                                             "orthogonal", self.args.ent_l2_norm)
                self.onto_triple_loss = hier_limited_loss(onto_phs, onto_prs, onto_pts, onto_nhs, onto_nrs, onto_nts,
                                                          self.args.pos_margin, self.args.neg_margin,
                                                          self.args.loss_norm, balance=self.args.neg_margin_balance,
                                                          weights=self.hier_weights, bias=self.hier_bias)
            elif self.args.onto_training_method == 'linear':
                self.onto_triple_loss = hier_limited_loss(onto_phs, onto_prs, onto_pts, onto_nhs, onto_nrs, onto_nts,
                                                          self.args.pos_margin, self.args.neg_margin,
                                                          self.args.loss_norm, balance=self.args.neg_margin_balance,
                                                          weights=None, bias=None)
            self.onto_triple_optimizer = generate_optimizer(self.onto_triple_loss, self.args.learning_rate,
                                                            opt=self.args.optimizer)

    def _define_ent2onto_mapping_graph(self):
        with tf.name_scope('cv_link_placeholder'):
            self.pos_ent_link = tf.placeholder(tf.int32, shape=[None])
            self.pos_onto_link = tf.placeholder(tf.int32, shape=[None])
            self.neg_ent_link = tf.placeholder(tf.int32, shape=[None])
            self.neg_onto_link = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('cv_link_lookup'):
            pos_ent_link = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.ent_embeds, self.pos_ent_link), 1)
            pos_onto_link = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.onto_ent_embeds, self.pos_onto_link), 1)
            neg_ent_link = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.ent_embeds, self.neg_ent_link), 1)
            neg_onto_link = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.onto_ent_embeds, self.neg_onto_link), 1)
        with tf.name_scope('cv_link_loss'):
            self.cvlink_weights, self.cvlink_bias = init_mlp(
                [self.args.dim, self.args.onto_dim], 'cv_link_weights', 'cv_link_bias', 'orthogonal', self.args.ent_l2_norm)
            self.ent2onto_mapping_loss = ent2onto_limited_loss(
                pos_ent_link, pos_onto_link, neg_ent_link, neg_onto_link, self.args.pos_margin, self.args.neg_margin,
                self.args.loss_norm, balance=self.args.neg_margin_balance, weights=self.cvlink_weights, bias=self.cvlink_bias)
            self.ent2onto_mapping_optimizer = generate_optimizer(self.ent2onto_mapping_loss,
                                                                 self.args.cv_link_lr,
                                                                 opt=self.args.optimizer)

    def _define_onto_likelihood_graph(self):
        self.ontologies1 = tf.placeholder(tf.int32, shape=[None])
        self.ontologies2 = tf.placeholder(tf.int32, shape=[None])
        dim = self.kgs.onto_entities_num
        dim1 = self.args.likelihood_slice
        self.onto_likelihood_mat = tf.placeholder(tf.float32, shape=[dim1, dim])
        ent1_embed = tf.nn.embedding_lookup(self.onto_ent_embeds, self.ontologies1)
        ent2_embed = tf.nn.embedding_lookup(self.onto_ent_embeds, self.ontologies2)
        mat = tf.log(tf.sigmoid(tf.matmul(ent1_embed, ent2_embed, transpose_b=True)))
        self.onto_likelihood_loss = self.args.sigma * (-tf.reduce_sum(tf.multiply(mat, self.onto_likelihood_mat)))
        self.onto_likelihood_optimizer = generate_optimizer(self.onto_likelihood_loss,
                                                            self.args.likelihood_lr,
                                                            opt=self.args.optimizer)

    '''
        seed_ontologies1:
        这里每个实体的class都是一个class path集合，例如“artist”-->[artist,person,agent,thing]
        seed_mask1:
        class path的掩码权重，通过参数delcay_weight(dw)控制权重的衰减速度，每个class的权重为[1,1*dw,1*dw^2,1*dw^3,...]
        如果delcay_weight=0，只考虑实体的单个class，即[artist,person,agent,thing]的掩码为[1,0,0,0]
        如果delcay_weight=1，则每个类别的权重都相同，即[artist,person,agent,thing]的掩码为[0.25,0.25,0.25,0.25]
        ops: seed_ontologies1 * seed_mask1即对class path embedding的加权求和，目前实验中将delcay_weight设置为0。
    '''

    def _define_mapping_graph(self, fusion_way=None):
        with tf.name_scope('seed_links_placeholder'):
            self.seed_entities1 = tf.placeholder(tf.int32, shape=[None])
            self.seed_entities2 = tf.placeholder(tf.int32, shape=[None])
            self.seed_neg_entities1 = tf.placeholder(tf.int32, shape=[None])
            self.seed_neg_entities2 = tf.placeholder(tf.int32, shape=[None])
            self.seed_ontologies1 = tf.placeholder(tf.int32, shape=[None, self.kgs.class_max_depth])
            self.seed_ontologies2 = tf.placeholder(tf.int32, shape=[None, self.kgs.class_max_depth])
            self.seed_mask1 = tf.placeholder(tf.float32, shape=[None, self.kgs.class_max_depth])
            self.seed_mask2 = tf.placeholder(tf.float32, shape=[None, self.kgs.class_max_depth])
        with tf.name_scope('seed_links_lookup'):
            ent1 = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.ent_embeds, self.seed_entities1), 1)
            ent2 = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.ent_embeds, self.seed_entities2), 1)
            neg_ent1 = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.ent_embeds, self.seed_neg_entities1), 1)
            neg_ent2 = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.ent_embeds, self.seed_neg_entities2), 1)
            onto1 = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.onto_ent_embeds, self.seed_ontologies1), 2)
            onto2 = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.onto_ent_embeds, self.seed_ontologies2), 2)
            # 伪本体对齐：||f(e1)-f(e2)||
            pseudo_onto1 = tf.nn.l2_normalize(tf.nn.tanh(tf.matmul(ent1, self.cvlink_weights) + self.cvlink_bias), 1)
            pseudo_onto2 = tf.nn.l2_normalize(tf.nn.tanh(tf.matmul(ent2, self.cvlink_weights) + self.cvlink_bias), 1)
            # class path路径加权求和
            seed_mask1 = tf.expand_dims(self.seed_mask1, axis=-1)
            seed_mask2 = tf.expand_dims(self.seed_mask2, axis=-1)
            weight_onto1 = tf.reduce_sum(tf.multiply(onto1, seed_mask1), axis=1)
            weight_onto2 = tf.reduce_sum(tf.multiply(onto2, seed_mask2), axis=1)

        with tf.name_scope('mapping_loss'):
            if fusion_way is None:
                self.mapping_loss = self.args.alpha * mapping_limit_loss(
                    ent1, ent2, neg_ent1, neg_ent2,
                    self.mapping_mat, self.args.pos_margin, self.args.neg_margin,
                    balance=self.args.neg_margin_balance, eye=self.eye_mat)
            elif fusion_way == 'add':
                ent_align_loss = self.args.alpha * mapping_limit_loss(
                    ent1, ent2, neg_ent1, neg_ent2,
                    self.mapping_mat, self.args.pos_margin, self.args.neg_margin,
                    balance=self.args.neg_margin_balance, eye=self.eye_mat)
                onto_align_loss = self.args.gamma * calibration_loss(weight_onto1, weight_onto2)
                pseudo_onto_algin_loss = self.args.beta * calibration_loss(pseudo_onto1, pseudo_onto2)
                self.mapping_loss = ent_align_loss + onto_align_loss + pseudo_onto_algin_loss
            else:
                raise Exception("unvaild fusion way.")
            self.mapping_optimizer = generate_optimizer(self.mapping_loss, self.args.learning_rate,
                                                        opt=self.args.optimizer)

    def onto_likelihood(self, training_epochs):
        onto_num = self.kgs.onto_entities_num
        likelihood_mat = self.kgs.onto_mat   # 根据发掘的对齐种子构建实体-实体矩阵
        likelihood_fetches = {
            "onto_likelihood_loss": self.onto_likelihood_loss,
            "onto_likelihood_op": self.onto_likelihood_optimizer}
        steps = onto_num // self.args.likelihood_slice
        ll = list(range(onto_num))
        for i in range(training_epochs):
            t = time.time()
            onto_likelihood_loss = 0.0
            for step in range(steps):
                idx = random.sample(ll, self.args.likelihood_slice)
                likelihood_feed_dict = {self.ontologies1: idx,
                                        self.ontologies2: ll,
                                        self.onto_likelihood_mat: likelihood_mat[idx, :]}
                vals = self.session.run(fetches=likelihood_fetches, feed_dict=likelihood_feed_dict)
                onto_likelihood_loss += vals["onto_likelihood_loss"]
            onto_likelihood_loss /= onto_num
            print("onto_likelihood_loss = {:.3f}, time = {:.3f} s".format(onto_likelihood_loss, time.time() - t))

    ######################################################
    ################### 训练每个模块    ####################
    ######################################################

    def launch_training_k_epo(self, iter, iter_nums, triple_steps, steps_tasks, training_batch_queue, neighbors1,
                              neighbors2):
        for i in range(1, iter_nums + 1):
            epoch = (iter - 1) * iter_nums + i
            self.launch_triple_training_1epo(epoch, triple_steps, steps_tasks, training_batch_queue, neighbors1,
                                             neighbors2)

    def launch_triple_training_1epo(self, epoch, triple_steps, steps_tasks, batch_queue, neighbors1, neighbors2):
        start = time.time()
        for steps_task in steps_tasks:
            mp.Process(target=bat.generate_relation_triple_batch_queue,
                       args=(self.kgs.kg1.relation_triples_list, self.kgs.kg2.relation_triples_list,
                             self.kgs.kg1.relation_triples_set, self.kgs.kg2.relation_triples_set,
                             self.kgs.kg1.entities_list, self.kgs.kg2.entities_list,
                             self.args.batch_size, steps_task,
                             batch_queue, neighbors1, neighbors2, self.args.neg_triple_num)).start()
        epoch_loss = 0
        trained_samples_num = 0
        for i in range(triple_steps):
            batch_pos, batch_neg = batch_queue.get()
            batch_loss, _ = self.session.run(fetches=[self.triple_loss, self.triple_optimizer],
                                             feed_dict={self.pos_hs: [x[0] for x in batch_pos],
                                                        self.pos_rs: [x[1] for x in batch_pos],
                                                        self.pos_ts: [x[2] for x in batch_pos],
                                                        self.neg_hs: [x[0] for x in batch_neg],
                                                        self.neg_rs: [x[1] for x in batch_neg],
                                                        self.neg_ts: [x[2] for x in batch_neg]})
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        random.shuffle(self.kgs.kg1.relation_triples_list)
        random.shuffle(self.kgs.kg2.relation_triples_list)
        print('epoch {}, avg. triple loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    def launch_onto_training_k_epo(self, iter, iter_nums, triple_steps, steps_tasks, training_batch_queue,
                                   neighbors=None, prob_dict=None):
        for i in range(1, iter_nums + 1):
            epoch = (iter - 1) * iter_nums + i
            self.launch_onto_triple_training_1epo(epoch, triple_steps, steps_tasks, training_batch_queue, neighbors, prob_dict)

    def launch_onto_triple_training_1epo(self, epoch, triple_steps, steps_tasks, batch_queue,
                                         neighbors=None, prob_dict=None):
        start = time.time()
        for steps_task in steps_tasks:
            mp.Process(target=bat.generate_onto_relation_triple_batch_queue,
                       args=(self.kgs.onto_kg.relation_triples_list,
                             self.kgs.onto_kg.relation_triples_set,
                             self.kgs.onto_kg.entities_list,
                             self.args.batch_size, steps_task,
                             batch_queue, neighbors, self.args.neg_triple_num, prob_dict)).start()
        epoch_loss = 0
        trained_samples_num = 0
        for i in range(triple_steps):
            batch_pos, batch_neg = batch_queue.get()
            # print sample result
            # onto_id_ents = dict(zip(self.kgs.onto_kg.entities_id_dict.values(),
            #                         self.kgs.onto_kg.entities_id_dict.keys()))  # 键值对互换
            # print('print batch pos :')
            # for i in range(min(len(batch_pos),5)):
            #     print(onto_id_ents[batch_pos[i][0]],onto_id_ents[batch_pos[i][2]])
            # print('print neg pos :')
            # for i in range(min(len(batch_neg), 10)):
            #     print(onto_id_ents[batch_neg[i][0]], onto_id_ents[batch_neg[i][2]])
            batch_loss, _ = self.session.run(fetches=[self.onto_triple_loss, self.onto_triple_optimizer],
                                             feed_dict={self.onto_pos_hs: [x[0] for x in batch_pos],
                                                        self.onto_pos_rs: [x[1] for x in batch_pos],
                                                        self.onto_pos_ts: [x[2] for x in batch_pos],
                                                        self.onto_neg_hs: [x[0] for x in batch_neg],
                                                        self.onto_neg_rs: [x[1] for x in batch_neg],
                                                        self.onto_neg_ts: [x[2] for x in batch_neg]})
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        random.shuffle(self.kgs.onto_kg.relation_triples_list)
        print('epoch {}, avg. onto triple loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    def launch_ent2onto_mapping_training_k_epo(self, iter, iter_nums, triple_steps, steps_tasks, training_batch_queue,
                                               neighbors=None, prob_dict=None):
        for i in range(1, iter_nums + 1):
            epoch = (iter - 1) * iter_nums + i
            self.launch_ent2onto_mapping_training_1epo(
                epoch, triple_steps, steps_tasks, training_batch_queue, neighbors, prob_dict)

    def launch_ent2onto_mapping_training_1epo(self, epoch, triple_steps, steps_tasks,
                                              batch_queue, neighbors=None, prob_dict=None):
        start = time.time()
        for steps_task in steps_tasks:
            mp.Process(target=bat.generate_cross_view_link_batch_queue,
                       args=(self.kgs.kg1_cv_links, self.kgs.kg2_cv_links,
                             set(self.kgs.kg1_cv_links),
                             set(self.kgs.kg2_cv_links),
                             self.kgs.onto_kg.entities_list,
                             self.args.batch_size, steps_task,
                             batch_queue, neighbors, self.args.neg_triple_num, prob_dict)).start()
        epoch_loss = 0
        trained_samples_num = 0
        for i in range(triple_steps):
            batch_pos, batch_neg = batch_queue.get()
            # onto_id_ents = dict(zip(self.kgs.onto_kg.entities_id_dict.values(),
            #                         self.kgs.onto_kg.entities_id_dict.keys()))  # 键值对互换
            # kg1_id_ents = dict(
            #     zip(self.kgs.kg1.entities_id_dict.values(), self.kgs.kg1.entities_id_dict.keys()))  # 键值对互换
            # kg2_id_ents = dict(
            #     zip(self.kgs.kg2.entities_id_dict.values(), self.kgs.kg2.entities_id_dict.keys()))  # 键值对互换
            # print('print batch pos :')
            # for i in range(min(len(batch_pos),5)):
            #     print(batch_pos[i])
            #     if batch_pos[i][0] in kg1_id_ents:
            #         print(kg1_id_ents[batch_pos[i][0]],onto_id_ents[batch_pos[i][1]])
            #     if batch_pos[i][0] in kg2_id_ents:
            #         print(kg2_id_ents[batch_pos[i][0]], onto_id_ents[batch_pos[i][1]])
            # print('print batch pos :')
            # for i in range(min(len(batch_neg), 10)):
            #     if batch_neg[i][0] in kg1_id_ents:
            #         print(kg1_id_ents[batch_neg[i][0]], onto_id_ents[batch_neg[i][1]])
            #     if batch_neg[i][0] in kg2_id_ents:
            #         print(kg2_id_ents[batch_neg[i][0]], onto_id_ents[batch_neg[i][1]])

            batch_loss, _ = self.session.run(fetches=[self.ent2onto_mapping_loss, self.ent2onto_mapping_optimizer],
                                             feed_dict={self.pos_ent_link: [x[0] for x in batch_pos],
                                                        self.pos_onto_link: [x[1] for x in batch_pos],
                                                        self.neg_ent_link: [x[0] for x in batch_neg],
                                                        self.neg_onto_link: [x[1] for x in batch_neg]})
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        random.shuffle(self.kgs.kg1_cv_links)
        random.shuffle(self.kgs.kg2_cv_links)
        print('epoch {}, avg. cv link mapping loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    # def launch_training_1epo(self, epoch, triple_steps, steps_tasks, training_batch_queue, neighbors1, neighbors2):
    #     self.launch_triple_training_1epo(epoch, triple_steps, steps_tasks, training_batch_queue, neighbors1, neighbors2)
    #     self.launch_mapping_training_1epo(epoch, triple_steps)

    def launch_mapping_training_k_epo(self, iter, iter_nums, triple_steps):
        for i in range(1, iter_nums + 1):
            epoch = (iter - 1) * iter_nums + i
            self.launch_mapping_training_1epo(epoch, triple_steps)

    def launch_mapping_training_1epo(self, epoch, triple_steps):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        for i in range(triple_steps):
            # train_links是一个列表，每个元素是一个元组
            # 采样小部分对齐实体对训练
            links_batch = random.sample(self.kgs.train_links, len(self.kgs.train_links) // triple_steps)
            # 生成对齐种子的负样本
            neg_links_batch = bat.generate_neg_seed_fast(links_batch, self.kgs.ent2onto_dict,
                                                         self.kgs.onto2ent_dict1, self.kgs.onto2ent_dict2,
                                                         self.kgs.useful_entities_list1, self.kgs.useful_entities_list2,
                                                         self.args.neg_triple_num,
                                                         method=self.args.alignment_neg_sampling)

            # print sample result
            # id_ent_dict1 = dict(zip(self.kgs.kg1.entities_id_dict.values(), self.kgs.kg1.entities_id_dict.keys()))  # 键值对互换
            # id_ent_dict2 = dict(
            #     zip(self.kgs.kg2.entities_id_dict.values(), self.kgs.kg2.entities_id_dict.keys()))  # 键值对互换
            # print('print batch pos :')
            # for i in range(min(len(links_batch),5)):
            #     print(id_ent_dict1[links_batch[i][0]],id_ent_dict2[links_batch[i][1]])
            # print('print neg pos :')
            # for i in range(min(len(neg_links_batch), 10)):
            #     print(id_ent_dict1[neg_links_batch[i][0]], id_ent_dict2[neg_links_batch[i][1]])
            # raise Exception('Stop')
            seed_onto_id1 = [self.kgs.ent2onto_dict[x[0]] for x in links_batch]
            seed_onto_path1 = [self.kgs.class_path_matrix[id] for id in seed_onto_id1]
            seed_onto_mask1 = [self.kgs.mask_path_matrix[id] for id in seed_onto_id1]
            seed_onto_id2 = [self.kgs.ent2onto_dict[x[1]] for x in links_batch]
            seed_onto_path2 = [self.kgs.class_path_matrix[id] for id in seed_onto_id2]
            seed_onto_mask2 = [self.kgs.mask_path_matrix[id] for id in seed_onto_id2]
            batch_loss, _ = self.session.run(fetches=[self.mapping_loss, self.mapping_optimizer],
                                             feed_dict={self.seed_entities1: [x[0] for x in links_batch],
                                                        self.seed_entities2: [x[1] for x in links_batch],
                                                        self.seed_neg_entities1: [x[0] for x in neg_links_batch],
                                                        self.seed_neg_entities2: [x[1] for x in neg_links_batch],
                                                        self.seed_ontologies1: seed_onto_path1,
                                                        self.seed_ontologies2: seed_onto_path2,
                                                        self.seed_mask1: seed_onto_mask1,
                                                        self.seed_mask2: seed_onto_mask2})
            epoch_loss += batch_loss
            trained_samples_num += len(links_batch)
        epoch_loss /= trained_samples_num
        print('epoch {}, avg. mapping loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    ######################################################
    ################### train 函数    ####################
    ######################################################

    def run(self):
        t = time.time()
        # 实体训练参数
        triples_num = self.kgs.kg1.relation_triples_num + self.kgs.kg2.relation_triples_num
        triple_steps = int(math.ceil(triples_num / self.args.batch_size))  # 每个epoch所需的step
        steps_tasks = task_divide(list(range(triple_steps)), self.args.batch_threads_num)  # 将step划分，交给多个线程处理
        # 本体训练参数
        onto_triples_num = self.kgs.onto_kg.relation_triples_num  # 本体三元组个数
        onto_triple_steps = int(math.ceil(onto_triples_num / self.args.onto_batch_size))  # 每个epoch所需的step
        onto_steps_tasks = task_divide(list(range(onto_triple_steps)), self.args.batch_threads_num)  # 将step划分，交给多个线程处理
        # membership link训练参数
        cv_links_num = len(self.kgs.cv_links)  # 本体三元组个数
        cv_links_steps = int(math.ceil(cv_links_num / self.args.cvlink_batch_size))  # 每个epoch所需的step
        cvlinks_steps_tasks = task_divide(list(range(cv_links_steps)), self.args.batch_threads_num)  # 将step划分，交给多个线程处理

        # 本体负采样概率矩阵
        if self.args.onto_neg_sampling == "prob_based":
            onto_prob_dict = sample.softmax((1 - self.kgs.onto_mat))
        else:
            onto_prob_dict = None
        # 实体负采样概率矩阵
        # entities_list = self.kgs.kg1.entities_list + self.kgs.kg2.entities_list
        # ent_prob_dict = sample.generate_ent_prob_dict(self.kgs.ent2onto_dict,entities_list,self.kgs.onto_mat)

        manager = mp.Manager()
        training_batch_queue = manager.Queue()

        neighbors1, neighbors2 = None, None
        sub_num = self.args.sub_epoch
        iter_nums = self.args.max_epoch // sub_num
        print('iter nums : {}'.format(iter_nums))
        for i in range(1, iter_nums + 1):
            print("\niteration", i)
            # 训练实体三元组
            self.launch_training_k_epo(i, sub_num, triple_steps, steps_tasks, training_batch_queue, neighbors1,
                                       neighbors2)
            # 训练本体三元组
            self.launch_onto_training_k_epo(i, sub_num, onto_triple_steps, onto_steps_tasks, training_batch_queue,
                                            prob_dict=onto_prob_dict)
            # 训练冲突矩阵似然Loss
            if self.args.likelihood_vaild:
                self.onto_likelihood(training_epochs=self.args.onto_likelihood_epochs)
            # 训练membership link
            self.launch_ent2onto_mapping_training_k_epo(i, sub_num, cv_links_steps, cvlinks_steps_tasks,
                                                        training_batch_queue,
                                                        prob_dict=None)
            # 训练对齐loss
            self.launch_mapping_training_k_epo(i, self.args.align_sub_epoch, triple_steps)

            if i * sub_num >= self.args.start_valid and i % self.args.eval_freq == 0:
                flag = self.valid(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop or i == iter_nums:
                    break

            if self.args.neg_sampling == 'truncated':
                t1 = time.time()
                assert 0.0 < self.args.truncated_epsilon < 1.0
                neighbors_num1 = int((1 - self.args.truncated_epsilon) * self.kgs.kg1.entities_num)
                neighbors_num2 = int((1 - self.args.truncated_epsilon) * self.kgs.kg2.entities_num)
                if neighbors1 is not None:
                    del neighbors1, neighbors2
                gc.collect()
                neighbors1 = bat.generate_neighbours(self.eval_kg1_useful_ent_embeddings(),
                                                     self.kgs.useful_entities_list1,
                                                     neighbors_num1, self.args.batch_threads_num)
                neighbors2 = bat.generate_neighbours(self.eval_kg2_useful_ent_embeddings(),
                                                     self.kgs.useful_entities_list2,
                                                     neighbors_num2, self.args.batch_threads_num)
                ent_num = len(self.kgs.kg1.entities_list) + len(self.kgs.kg2.entities_list)
                print("\ngenerating neighbors of {} entities costs {:.3f} s.".format(ent_num, time.time() - t1))
                gc.collect()
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))

    def _eval_valid_embeddings(self):

        if len(self.kgs.valid_links) > 0:
            embeds1 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.valid_entities1).eval(session=self.session)
            embeds2 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.valid_entities2 + self.kgs.test_entities2).eval(
                session=self.session)
            # 求的是路径
            seed_onto_path1 = [self.kgs.class_path_matrix[id] for id in self.kgs.valid_ontologies1]
            seed_onto_mask1 = [self.kgs.mask_path_matrix[id] for id in self.kgs.valid_ontologies1]
            vaild_test_ontologies = self.kgs.valid_ontologies2 + self.kgs.test_ontologies2
            seed_onto_path2 = [self.kgs.class_path_matrix[id] for id in vaild_test_ontologies]
            seed_onto_mask2 = [self.kgs.mask_path_matrix[id] for id in vaild_test_ontologies]

            onto_path_embeds1 = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.onto_ent_embeds, seed_onto_path1), 1).eval(
                session=self.session) if self.onto_ent_embeds is not None else None
            onto_path_embeds2 = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.onto_ent_embeds, seed_onto_path2), 1).eval(
                session=self.session) if self.onto_ent_embeds is not None else None
        else:
            embeds1 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.test_entities1).eval(session=self.session)
            embeds2 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.test_entities2).eval(session=self.session)
            # 求的是路径
            seed_onto_path1 = [self.kgs.class_path_matrix[id] for id in self.kgs.test_ontologies1]
            seed_onto_mask1 = [self.kgs.mask_path_matrix[id] for id in self.kgs.test_ontologies1]
            seed_onto_path2 = [self.kgs.class_path_matrix[id] for id in self.kgs.test_ontologies2]
            seed_onto_mask2 = [self.kgs.mask_path_matrix[id] for id in self.kgs.test_ontologies2]
            # print(seed_onto_path1)
            # print(seed_onto_mask1)
            onto_path_embeds1 = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.onto_ent_embeds, seed_onto_path1), 1).eval(
                session=self.session) if self.onto_ent_embeds is not None else None
            onto_path_embeds2 = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.onto_ent_embeds, seed_onto_path2), 1).eval(
                session=self.session) if self.onto_ent_embeds is not None else None
        seed_mask1 = np.expand_dims(seed_onto_mask1, axis=-1)
        seed_mask2 = np.expand_dims(seed_onto_mask2, axis=-1)
        # print(onto_path_embeds1.shape,seed_mask1.shape)
        weight_onto_embeds1 = np.sum(np.multiply(onto_path_embeds1, seed_mask1), axis=1)
        weight_onto_embeds2 = np.sum(np.multiply(onto_path_embeds2, seed_mask2), axis=1)
        mapping = self.mapping_mat.eval(session=self.session) if self.mapping_mat is not None else None
        cvlink_weights = self.cvlink_weights.eval(session=self.session) if self.cvlink_weights is not None else None
        cvlink_bias = self.cvlink_bias.eval(session=self.session) if self.cvlink_bias is not None else None
        return embeds1, embeds2, weight_onto_embeds1, weight_onto_embeds2, mapping, cvlink_weights, cvlink_bias

    def _eval_test_embeddings(self):
        embeds1 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.test_entities1).eval(session=self.session)
        embeds2 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.test_entities2).eval(session=self.session)
        # 求的是路径
        seed_onto_path1 = [self.kgs.class_path_matrix[id] for id in self.kgs.test_ontologies1]
        seed_onto_mask1 = [self.kgs.mask_path_matrix[id] for id in self.kgs.test_ontologies1]
        seed_onto_path2 = [self.kgs.class_path_matrix[id] for id in self.kgs.test_ontologies2]
        seed_onto_mask2 = [self.kgs.mask_path_matrix[id] for id in self.kgs.test_ontologies2]
        onto_path_embeds1 = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.onto_ent_embeds, seed_onto_path1), 1).eval(
            session=self.session) if self.onto_ent_embeds is not None else None
        onto_path_embeds2 = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.onto_ent_embeds, seed_onto_path2), 1).eval(
            session=self.session) if self.onto_ent_embeds is not None else None
        seed_mask1 = np.expand_dims(seed_onto_mask1, axis=-1)
        seed_mask2 = np.expand_dims(seed_onto_mask2, axis=-1)
        weight_onto_embeds1 = np.sum(np.multiply(onto_path_embeds1, seed_mask1), axis=1)
        weight_onto_embeds2 = np.sum(np.multiply(onto_path_embeds2, seed_mask2), axis=1)
        mapping = self.mapping_mat.eval(session=self.session) if self.mapping_mat is not None else None
        cvlink_weights = self.cvlink_weights.eval(session=self.session) if self.cvlink_weights is not None else None
        cvlink_bias = self.cvlink_bias.eval(session=self.session) if self.cvlink_bias is not None else None
        return embeds1, embeds2, weight_onto_embeds1, weight_onto_embeds2, mapping, cvlink_weights, cvlink_bias

    def valid(self, stop_metric):
        embeds1, embeds2, onto_embeds1, onto_embeds2, mapping, cvlink_weights, cvlink_bias = self._eval_valid_embeddings()

        # 测试的时候只考虑实体信息
        # eval_metric : The distance metric to use. It can be 'cosine', 'euclidean' or 'inner'.
        print('alignment only with entity embedding : ')
        _, _ = valid(embeds1, embeds2, mapping, self.args.top_k,
                     self.args.test_threads_num, metric=self.args.eval_metric,
                     normalize=self.args.eval_norm, csls_k=0, accurate=False, type_mat=None,
                     type_weight=self.args.gamma)
        # 保持训练和测试的fusion way一致
        print('alignment with weighted sum of entity embedding and class embedding : ')
        fusion_embeds1, fusion_embeds2, type_matrix = self.fusion_embeddings(
            embeds1, embeds2, onto_embeds1, onto_embeds2,
            mapping=mapping, fusion_way=self.args.fusion_way, onto_weight=self.args.gamma)
        # eval_metric : The distance metric to use. It can be 'cosine', 'euclidean' or 'inner'.
        hits1_12, mrr_12 = valid(fusion_embeds1, fusion_embeds2, None, self.args.top_k,
                                 self.args.test_threads_num, metric=self.args.eval_metric,
                                 normalize=self.args.eval_norm, csls_k=0, accurate=False,
                                 type_mat=type_matrix, type_weight=self.args.gamma)
        return hits1_12 if stop_metric == 'hits1' else mrr_12

    def test(self, save=True):
        # type_matrix = None
        embeds1, embeds2, onto_embeds1, onto_embeds2, mapping, cvlink_weights, cvlink_bias = self._eval_test_embeddings()
        print('alignment only with entity embedding : ')
        # 测试的时候只考虑实体信息
        _, _, _ = test(embeds1, embeds2, mapping, self.args.top_k,
                       self.args.test_threads_num, metric=self.args.eval_metric,
                       normalize=self.args.eval_norm, csls_k=0, accurate=True, type_mat=None,
                       type_weight=self.args.gamma)
        print('alignment with weighted sum of entity embedding and class embedding : ')
        # 保持训练和测试的fusion way一致
        fusion_embeds1, fusion_embeds2, type_matrix = self.fusion_embeddings(embeds1, embeds2, onto_embeds1,
                                                                             onto_embeds2,
                                                                             mapping=mapping,
                                                                             fusion_way=self.args.fusion_way,
                                                                             onto_weight=self.args.gamma)
        rest_12, _, _ = test(fusion_embeds1, fusion_embeds2, None, self.args.top_k, self.args.test_threads_num,
                             metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0,
                             accurate=True, type_mat=type_matrix, type_weight=self.args.gamma)

        print('Using CSLS & alignment with weighted sum of entity embedding and class embedding : ')
        test(fusion_embeds1, fusion_embeds2, None, self.args.top_k, self.args.test_threads_num,
             metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=self.args.csls,
             accurate=True, type_mat=type_matrix, type_weight=self.args.gamma)

        # # 使用学习得到的本体作为相似度
        # print('using pesudo ontology alignment  : ')
        # pesudo_onto_embeds1 = np.tanh(np.matmul(embeds1, cvlink_weights) + cvlink_bias)
        # pesudo_onto_embeds2 = np.tanh(np.matmul(embeds2, cvlink_weights) + cvlink_bias)
        # pesudo_type_mat = sim(pesudo_onto_embeds1, pesudo_onto_embeds2, metric='cosine', normalize=True)
        # _, _,_ = test(fusion_embeds1, fusion_embeds2, None, self.args.top_k, self.args.test_threads_num,
        #                          metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0,
        #                          accurate=True,type_mat=pesudo_type_mat,type_weight=self.args.gamma)
        # test(fusion_embeds1, fusion_embeds2, None, self.args.top_k, self.args.test_threads_num,
        #      metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=self.args.csls,
        #      accurate=True,type_mat=pesudo_type_mat,type_weight=self.args.gamma)

        #####################  枚举type weight #########################
        # weight_lists = [0,0.1,0.2,0.4,0.5,0.6]
        # for gt in weight_lists:
        #     print('ontology weight : {}'.format(gt))
        #     # print('grouth truth weight : {}, pesudo label weight : {} '.format(gt,0.5-gt))
        #     # weight_type_matrix = gt * type_matrix + (0.5-gt) * pesudo_type_mat
        #     rest_12, _, _ = test(fusion_embeds1, fusion_embeds2, None, self.args.top_k, self.args.test_threads_num,
        #                              metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0,
        #                              accurate=True, type_mat=type_matrix, type_weight=gt, enum_weight=True)
        #     test(fusion_embeds1, fusion_embeds2, None, self.args.top_k, self.args.test_threads_num,
        #              metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=self.args.csls,
        #              accurate=True, type_mat=type_matrix, type_weight=gt, enum_weight=True)

        if save:
            ent_ids_rest_12 = [(self.kgs.test_entities1[i], self.kgs.test_entities2[j]) for i, j in rest_12]
            rd.save_results(self.out_folder, ent_ids_rest_12)

    def fusion_embeddings(self, ent1, ent2, onto1, onto2, mapping=None, fusion_way=None, onto_weight=1.0):
        if fusion_way is None:
            ent1_mapped = np.matmul(ent1, mapping)
            type_mat = sim(onto1, onto2, metric='cosine', normalize=True)
            return ent1_mapped, ent2, type_mat
        elif fusion_way == 'add':
            ent1_mapped = np.matmul(ent1, mapping)
            type_mat = sim(onto1, onto2, metric='cosine', normalize=True)
            return ent1_mapped, ent2, type_mat
        # elif fusion_way == 'mapping_concat':
        #     ent1_mapped = np.matmul(ent1, mapping)
        #     concat1 = np.concatenate([ent1_mapped,onto_weight*onto1],axis=1)
        #     concat2 = np.concatenate([ent2,onto_weight*onto2],axis=1)
        #     return concat1,concat2,None
        # elif fusion_way == 'calibration_concat':
        #     concat1 = np.concatenate([ent1, onto_weight*onto1], axis=1)
        #     concat2 = np.concatenate([ent2, onto_weight*onto2], axis=1)
        #     return concat1, concat2, None
