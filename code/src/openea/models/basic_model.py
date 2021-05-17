import math
import multiprocessing as mp
import random
import time
import gc
import sys
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import openea.modules.load.read as rd
import openea.modules.train.batch as bat
from openea.modules.finding.evaluation import valid
from openea.modules.finding.evaluation import test
from openea.modules.finding.evaluation import early_stop
from openea.modules.utils.util import generate_out_folder
from openea.modules.utils.util import task_divide
from openea.modules.base.initializers import init_embeddings
from openea.modules.base.losses import get_loss_func
from openea.modules.base.optimizers import generate_optimizer
from openea.modules.base.mapping import add_mapping_variables
from openea.modules.base.mapping import add_onto_mapping_variables
from openea.modules.base.mapping import add_mapping_module
from openea.modules.base.mapping import add_mapping_module_with_ontoAlign
from openea.modules.base.mapping import add_onto_mapping_module
from openea.modules.utils.check import check_disjoint
from openea.modules.utils.check import load_type_relation
from openea.modules.utils.check import check_type_coherence
from openea.modules.utils import inference as infer
from openea.modules.finding.alignment import stable_alignment


sys.path.append("../")


class BasicModel:

    def set_kgs(self, kgs):
        self.kgs = kgs

    def set_args(self, args):
        self.args = args
        self.out_folder = generate_out_folder(self.args.output,
                                              self.args.training_data,
                                              self.args.dataset_division,
                                              self.__class__.__name__)

    def init(self):
        # need to be overwriten
        pass

    def __init__(self):

        self.out_folder = None
        self.args = None
        self.kgs = None

        self.session = None

        self.seed_entities1 = None
        self.seed_entities2 = None
        self.seed_ontologies1 = None
        self.seed_ontologies2 = None

        self.entities = None    # cv link中的实体
        self.ontologies = None  # cv link中的本体

        self.neg_ts = None
        self.neg_rs = None
        self.neg_hs = None
        self.pos_ts = None
        self.pos_rs = None
        self.pos_hs = None

        self.pos_onto_hs = None
        self.pos_onto_rs = None
        self.pos_onto_ts = None
        self.neg_onto_hs = None
        self.neg_onto_rs = None
        self.neg_onto_ts = None

        self.rel_embeds = None
        self.ent_embeds = None
        self.onto_rel_embeds = None
        self.onto_ent_embeds = None

        self.mapping_mat = None
        self.eye_mat = None
        self.onto_mapping_mat = None
        self.onto_eye_mat = None

        self.triple_optimizer = None
        self.triple_loss = None
        self.mapping_optimizer = None
        self.mapping_loss = None
        self.onto_mapping_optimizer = None
        self.onto_mapping_loss = None

        self.mapping_mat = None
        self.onto_mapping_mat = None

        self.flag1 = -1
        self.flag2 = -1
        self.early_stop = False

    def _define_variables(self):
        with tf.variable_scope('relational' + 'embeddings'):
            self.ent_embeds = init_embeddings([self.kgs.entities_num, self.args.dim], 'ent_embeds',
                                              self.args.init, self.args.ent_l2_norm)
            self.rel_embeds = init_embeddings([self.kgs.relations_num, self.args.dim], 'rel_embeds',
                                              self.args.init, self.args.rel_l2_norm)

    def _define_embed_graph(self):
        with tf.name_scope('triple_placeholder'):
            self.pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.pos_rs = tf.placeholder(tf.int32, shape=[None])
            self.pos_ts = tf.placeholder(tf.int32, shape=[None])
            self.neg_hs = tf.placeholder(tf.int32, shape=[None])
            self.neg_rs = tf.placeholder(tf.int32, shape=[None])
            self.neg_ts = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('triple_lookup'):
            phs = tf.nn.embedding_lookup(self.ent_embeds, self.pos_hs)
            prs = tf.nn.embedding_lookup(self.rel_embeds, self.pos_rs)
            pts = tf.nn.embedding_lookup(self.ent_embeds, self.pos_ts)
            nhs = tf.nn.embedding_lookup(self.ent_embeds, self.neg_hs)
            nrs = tf.nn.embedding_lookup(self.rel_embeds, self.neg_rs)
            nts = tf.nn.embedding_lookup(self.ent_embeds, self.neg_ts)
        with tf.name_scope('triple_loss'):
            self.triple_loss = get_loss_func(phs, prs, pts, nhs, nrs, nts, self.args)
            self.triple_optimizer = generate_optimizer(self.triple_loss, self.args.learning_rate,
                                                       opt=self.args.optimizer)

    def _define_mapping_variables(self):
        add_mapping_variables(self)

    def _define_onto_mapping_variables(self, method='linear'):
        add_onto_mapping_variables(self, method=method)

    def _define_mapping_graph(self, onto_vaild=False):
        if onto_vaild:
            add_mapping_module_with_ontoAlign(self)
        else:
            add_mapping_module(self)

    def _define_onto_mapping_graph(self, orthogonal=True, method='linear'):
        add_onto_mapping_module(self, orthogonal=orthogonal, method=method)

    def _eval_valid_embeddings(self):
        if len(self.kgs.valid_links) > 0:
            embeds1 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.valid_entities1).eval(session=self.session)
            embeds2 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.valid_entities2 + self.kgs.test_entities2).eval(
                session=self.session)  # 这里为啥要加上test_entities2的实体？
        else:
            embeds1 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.test_entities1).eval(session=self.session)
            embeds2 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.test_entities2).eval(session=self.session)
        mapping = self.mapping_mat.eval(session=self.session) if self.mapping_mat is not None else None
        return embeds1, embeds2, mapping

    def _eval_test_embeddings(self):
        embeds1 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.test_entities1).eval(session=self.session)
        embeds2 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.test_entities2).eval(session=self.session)
        mapping = self.mapping_mat.eval(session=self.session) if self.mapping_mat is not None else None
        return embeds1, embeds2, mapping

    def _eval_test_types(self, supervised_ratio=0, unsure_w=0.5, unify=1, check_version=0,
                         class_complement=False, class_path_truncation=-1, max_depth=4,
                         entropy_threshold=0.4, cal_by_sup=True):
        '''
        normal : True : 本体不对齐的实体对，统一成一个本体
        '''
        # save_file = h5py.File('data.h5','w')
        lang = os.path.split(self.args.training_data.rstrip('/'))[1]
        kg1_lang, kg2_lang = lang.split('_')
        uri_test_entities1 = [link[0] for link in self.kgs.uri_test_links]
        uri_test_entities2 = [link[1] for link in self.kgs.uri_test_links]

        # 生成本体矩阵
        type_relation = load_type_relation(self.args, blacket=True)
        check_type_coherence(type_relation, self.args, unsure_w=unsure_w, version=check_version)         # 生成本体矩阵
        onto_check_mat = rd.load_onto_check_mat(os.path.join(self.args.dataset_division, 'onto_check_mat.h5'))
        onto_mat = onto_check_mat['onto_mat']
        onto2id_dict = onto_check_mat['onto2id_dict']

        # class补全 & 路径截断
        sup_links = self.kgs.uri_train_links
        ent_type_dict1, ent_type_dict2 = infer.class_preprocess(
            folder=os.path.split(self.args.training_data.rstrip('/'))[0], lang=lang,
            supervised_links=sup_links, complement=class_complement,
            trunction=class_path_truncation, print_info=True, unify=unify)

        # 利用对齐种子校验mat
        if supervised_ratio > 0:
            total_uri_links = self.kgs.uri_train_links + self.kgs.uri_valid_links + self.kgs.uri_test_links
            supervised_links_num = int(len(total_uri_links) * supervised_ratio)
            sup_links = total_uri_links[:supervised_links_num]
        else:
            sup_links = self.kgs.uri_train_links
        print('supervised_links_num : ', len(sup_links))
        if cal_by_sup:
            for a, b in sup_links:
                atype, btype = ent_type_dict1[a], ent_type_dict2[b]
                atype_id, btype_id = onto2id_dict[atype], onto2id_dict[btype]
                flag = check_disjoint(type_relation['disjointwith'], atype, btype)
                if not flag:   # 过滤冲突的例子
                    onto_mat[atype_id][btype_id] = 1

        uri_test_types1 = [ent_type_dict1[ent] for ent in uri_test_entities1]
        uri_test_types2 = [ent_type_dict2[ent] for ent in uri_test_entities2]
        print('start checking entities types : ')
        type_matrix = np.zeros((len(uri_test_types1), len(uri_test_types2)))
        uri_test_types_id1, uri_test_types_id2 = [], []
        for i in range(len(uri_test_types1)):
            uri_test_types_id1.append(onto2id_dict[uri_test_types1[i]])
        for i in range(len(uri_test_types2)):
            uri_test_types_id2.append(onto2id_dict[uri_test_types2[i]])
        print(type(onto_mat))
        print(type(uri_test_types1), len(uri_test_types1))
        try:
            with tqdm(range(len(uri_test_types1)), ncols=150) as t:
                for i in t:
                    # type_id1 = onto2id_dict[uri_test_types1[i]]
                    for j in range(len(uri_test_types2)):
                        # type_id2 = onto2id_dict[uri_test_types2[j]]
                        # flag = check_type(onto_type,uri_test_types1[i],uri_test_types2[j])
                        # print(uri_test_types_id1[i],uri_test_types_id2[j])
                        type_matrix[i][j] = onto_mat[uri_test_types_id1[i]][uri_test_types_id2[j]]
                        # type_matrix[i][j] = 1
        except KeyboardInterrupt:
            t.close()
            raise
        t.close()

        # save_file.create_dataset('type_mat',data=type_matrix)
        # save_file.create_dataset('uri')
        print('check finished...')
        return type_matrix

    def valid(self, stop_metric):
        embeds1, embeds2, mapping = self._eval_valid_embeddings()
        # eval_metric : The distance metric to use. It can be 'cosine', 'euclidean' or 'inner'.
        hits1_12, mrr_12 = valid(embeds1, embeds2, mapping, self.args.top_k,
                                 self.args.test_threads_num, metric=self.args.eval_metric,
                                 normalize=self.args.eval_norm, csls_k=0, accurate=False)
        return hits1_12 if stop_metric == 'hits1' else mrr_12

    def test(self, save=True, onto_eval=False, type_weight=0.2):
        type_matrix = None

        embeds1, embeds2, mapping = self._eval_test_embeddings()

        if onto_eval:
            type_matrix = self._eval_test_types()
        rest_12, _, _ = test(embeds1, embeds2, mapping, self.args.top_k, self.args.test_threads_num,
                             metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0,
                             accurate=True, onto_eval=onto_eval, type_mat=type_matrix, type_weight=type_weight)
        test(embeds1, embeds2, mapping, self.args.top_k, self.args.test_threads_num,
             metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=self.args.csls,
             accurate=True, onto_eval=onto_eval, type_mat=type_matrix, type_weight=type_weight)
        if save:
            ent_ids_rest_12 = [(self.kgs.test_entities1[i], self.kgs.test_entities2[j]) for i, j in rest_12]
            rd.save_results(self.out_folder, ent_ids_rest_12)

    def test_with_types(self, save=True, onto_eval=False, weight_lists=None,
                        supervised_ratio=0, unsure_w=0, unify=1, check_version=0,
                        class_complement=False, class_path_truncation=-1,
                        entropy_threshold=0.4, cal_by_sup=True, csls=False):
        type_matrix = None
        if onto_eval:
            type_matrix = self._eval_test_types(
                supervised_ratio=supervised_ratio, unsure_w=unsure_w,
                unify=unify, check_version=check_version, class_complement=class_complement,
                class_path_truncation=class_path_truncation,
                entropy_threshold=entropy_threshold, cal_by_sup=cal_by_sup)
        embeds1, embeds2, mapping = self._eval_test_embeddings()
        for w in weight_lists:
            print('the type weight is {}'.format(w))
            rest_12, _, _ = test(embeds1, embeds2, mapping, self.args.top_k, self.args.test_threads_num,
                                 metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0,
                                 accurate=True, onto_eval=self.args.onto_eval, type_mat=type_matrix,
                                 type_weight=w)
            if csls:
                test(embeds1, embeds2, mapping, self.args.top_k, self.args.test_threads_num,
                     metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=self.args.csls,
                     accurate=True, onto_eval=onto_eval, type_mat=type_matrix, type_weight=w)
        # if save:
        #     ent_ids_rest_12 = [(self.kgs.test_entities1[i], self.kgs.test_entities2[j]) for i, j in rest_12]
        #     rd.save_results(self.out_folder, ent_ids_rest_12)

    def retest(self):
        dir = self.out_folder.split("/")
        new_dir = ""
        for i in range(len(dir) - 2):
            new_dir += (dir[i] + "/")
        exist_file = os.listdir(new_dir)
        new_dir = new_dir + exist_file[0] + "/"
        embeds = np.load(new_dir + "ent_embeds.npy")
        embeds1 = embeds[self.kgs.test_entities1]
        embeds2 = embeds[self.kgs.test_entities2]
        mapping = None

        print(self.__class__.__name__, type(self.__class__.__name__))
        if self.__class__.__name__ == "GCN_Align":
            print(self.__class__.__name__, "loads attr embeds")
            attr_embeds = np.load(new_dir + "attr_embeds.npy")
            attr_embeds1 = attr_embeds[self.kgs.test_entities1]
            attr_embeds2 = attr_embeds[self.kgs.test_entities2]
            embeds1 = np.concatenate([embeds1 * self.args.beta, attr_embeds1 * (1.0 - self.args.beta)], axis=1)
            embeds2 = np.concatenate([embeds2 * self.args.beta, attr_embeds2 * (1.0 - self.args.beta)], axis=1)

        # if self.__class__.__name__ == "MTransE" or self.__class__.__name__ == "SEA" or self.__class__.__name__ == "KDCoE":
        if os.path.exists(new_dir + "mapping_mat.npy"):
            print(self.__class__.__name__, "loads mapping mat")
            mapping = np.load(new_dir + "mapping_mat.npy")

        print("conventional test:")
        test(embeds1, embeds2, mapping, self.args.top_k, self.args.test_threads_num,
             metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0, accurate=True)
        print("conventional reversed test:")
        if mapping is not None:
            embeds1 = np.matmul(embeds1, mapping)
            test(embeds2, embeds1, None, self.args.top_k, self.args.test_threads_num,
                 metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0, accurate=True)
        else:
            test(embeds2, embeds1, mapping, self.args.top_k, self.args.test_threads_num,
                 metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0, accurate=True)
        print("stable test:")
        stable_alignment(embeds1, embeds2, self.args.eval_metric, self.args.eval_norm, csls_k=0,
                         nums_threads=self.args.test_threads_num)
        print("stable test with csls:")
        stable_alignment(embeds1, embeds2, self.args.eval_metric, self.args.eval_norm, csls_k=self.args.csls,
                         nums_threads=self.args.test_threads_num)

    def save(self):
        ent_embeds = self.ent_embeds.eval(session=self.session)
        rel_embeds = self.rel_embeds.eval(session=self.session)
        mapping_mat = self.mapping_mat.eval(session=self.session) if self.mapping_mat is not None else None
        rd.save_embeddings(self.out_folder, self.kgs, ent_embeds, rel_embeds, None, mapping_mat=mapping_mat)

    def eval_kg1_ent_embeddings(self):
        embeds = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.kg1.entities_list)
        return embeds.eval(session=self.session)

    def eval_kg2_ent_embeddings(self):
        embeds = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.kg2.entities_list)
        return embeds.eval(session=self.session)

    def eval_kg1_useful_ent_embeddings(self):
        embeds = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.useful_entities_list1)
        return embeds.eval(session=self.session)

    def eval_kg2_useful_ent_embeddings(self):
        embeds = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.useful_entities_list2)
        return embeds.eval(session=self.session)

    def launch_training_1epo(self, epoch, triple_steps, steps_tasks, training_batch_queue, neighbors1, neighbors2):
        self.launch_triple_training_1epo(epoch, triple_steps, steps_tasks, training_batch_queue, neighbors1, neighbors2)
        if self.args.alignment_module == 'mapping':
            self.launch_mapping_training_1epo(epoch, triple_steps)

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

    def launch_mapping_training_1epo(self, epoch, triple_steps):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        for i in range(triple_steps):
            links_batch = random.sample(self.kgs.train_links, len(self.kgs.train_links) // triple_steps)
            batch_loss, _ = self.session.run(fetches=[self.mapping_loss, self.mapping_optimizer],
                                             feed_dict={self.seed_entities1: [x[0] for x in links_batch],
                                                        self.seed_entities2: [x[1] for x in links_batch]})
            epoch_loss += batch_loss
            trained_samples_num += len(links_batch)
        epoch_loss /= trained_samples_num
        print('epoch {}, avg. mapping loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    def run(self):
        t = time.time()
        triples_num = self.kgs.kg1.relation_triples_num + self.kgs.kg2.relation_triples_num
        triple_steps = int(math.ceil(triples_num / self.args.batch_size))
        steps_tasks = task_divide(list(range(triple_steps)), self.args.batch_threads_num)
        manager = mp.Manager()
        training_batch_queue = manager.Queue()
        neighbors1, neighbors2 = None, None
        for i in range(1, self.args.max_epoch + 1):
            self.launch_training_1epo(i, triple_steps, steps_tasks, training_batch_queue, neighbors1, neighbors2)
            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                flag = self.valid(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop or i == self.args.max_epoch:
                    break
            if self.args.neg_sampling == 'truncated' and i % self.args.truncated_freq == 0:
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
