import os
import sys
import numpy as np
import random
from openea.modules.load.kg import KG
from openea.modules.load.read import generate_sharing_id
from openea.modules.load.read import generate_mapping_id
from openea.modules.load.read import uris_relation_triple_2ids
from openea.modules.load.read import uris_attribute_triple_2ids
from openea.modules.load.read import uris_pair_2ids
from openea.modules.load.read import generate_sup_relation_triples
from openea.modules.load.read import generate_sup_attribute_triples
from openea.modules.load.read import load_class_path
from openea.modules.load.read import generate_mapping_id_oneKG
from openea.modules.load.read import read_relation_triples
from openea.modules.load.read import read_attribute_triples
from openea.modules.load.read import read_entType_file
from openea.modules.load.read import read_links
from openea.modules.load.read import load_name_dicts
from openea.modules.load.read import generate_sup_cv_links
from openea.modules.utils.check import load_type_relation
from openea.modules.utils.check import check_type_coherence
import openea.modules.load.read as rd


sys.path.append("../")
MISS_URI = 'http://www.w3.org/2002/07/owl#Thing'
NEG_INF = -1000000000


class KGs:
    def __init__(self, kg1: KG, kg2: KG, train_links, test_links, valid_links=None, mode='mapping', ordered=True):
        if mode == "sharing":
            ent_ids1, ent_ids2 = generate_sharing_id(train_links, kg1.relation_triples_set, kg1.entities_set,
                                                     kg2.relation_triples_set, kg2.entities_set, ordered=ordered)
            rel_ids1, rel_ids2 = generate_sharing_id([], kg1.relation_triples_set, kg1.relations_set,
                                                     kg2.relation_triples_set, kg2.relations_set, ordered=ordered)
            attr_ids1, attr_ids2 = generate_sharing_id([], kg1.attribute_triples_set, kg1.attributes_set,
                                                       kg2.attribute_triples_set, kg2.attributes_set, ordered=ordered)
        else:
            # 所有的实体加在一起编号
            ent_ids1, ent_ids2 = generate_mapping_id(kg1.relation_triples_set, kg1.entities_set,
                                                     kg2.relation_triples_set, kg2.entities_set, ordered=ordered)
            rel_ids1, rel_ids2 = generate_mapping_id(kg1.relation_triples_set, kg1.relations_set,
                                                     kg2.relation_triples_set, kg2.relations_set, ordered=ordered)
            attr_ids1, attr_ids2 = generate_mapping_id(kg1.attribute_triples_set, kg1.attributes_set,
                                                       kg2.attribute_triples_set, kg2.attributes_set, ordered=ordered)

        # 关系三元组到id的转换
        id_relation_triples1 = uris_relation_triple_2ids(kg1.relation_triples_set, ent_ids1, rel_ids1)
        id_relation_triples2 = uris_relation_triple_2ids(kg2.relation_triples_set, ent_ids2, rel_ids2)

        # 属性三元组到id的转换
        id_attribute_triples1 = uris_attribute_triple_2ids(kg1.attribute_triples_set, ent_ids1, attr_ids1)
        id_attribute_triples2 = uris_attribute_triple_2ids(kg2.attribute_triples_set, ent_ids2, attr_ids2)

        self.uri_kg1 = kg1
        self.uri_kg2 = kg2

        kg1 = KG(id_relation_triples1, id_attribute_triples1)
        kg2 = KG(id_relation_triples2, id_attribute_triples2)
        kg1.set_id_dict(ent_ids1, rel_ids1, attr_ids1)
        kg2.set_id_dict(ent_ids2, rel_ids2, attr_ids2)

        self.uri_train_links = train_links
        self.uri_test_links = test_links
        self.train_links = uris_pair_2ids(self.uri_train_links, ent_ids1, ent_ids2)  # uri转id
        self.test_links = uris_pair_2ids(self.uri_test_links, ent_ids1, ent_ids2)
        self.train_entities1 = [link[0] for link in self.train_links]
        self.train_entities2 = [link[1] for link in self.train_links]
        self.test_entities1 = [link[0] for link in self.test_links]
        self.test_entities2 = [link[1] for link in self.test_links]

        # 替换包含对齐实体的三元组中的头实体或者尾实体，增强三元组个数
        if mode == 'swapping':
            sup_triples1, sup_triples2 = generate_sup_relation_triples(self.train_links,
                                                                       kg1.rt_dict, kg1.hr_dict,
                                                                       kg2.rt_dict, kg2.hr_dict)
            kg1.add_sup_relation_triples(sup_triples1)
            kg2.add_sup_relation_triples(sup_triples2)

            sup_triples1, sup_triples2 = generate_sup_attribute_triples(self.train_links, kg1.av_dict, kg2.av_dict)
            kg1.add_sup_attribute_triples(sup_triples1)
            kg2.add_sup_attribute_triples(sup_triples2)

        self.kg1 = kg1  # id编码的KG
        self.kg2 = kg2  # id编码的KG

        self.valid_links = list()
        self.valid_entities1 = list()
        self.valid_entities2 = list()
        # 验证集不为空
        if valid_links is not None:
            self.uri_valid_links = valid_links
            self.valid_links = uris_pair_2ids(self.uri_valid_links, ent_ids1, ent_ids2)
            self.valid_entities1 = [link[0] for link in self.valid_links]
            self.valid_entities2 = [link[1] for link in self.valid_links]

        self.useful_entities_list1 = self.train_entities1 + self.valid_entities1 + self.test_entities1
        self.useful_entities_list2 = self.train_entities2 + self.valid_entities2 + self.test_entities2

        self.entities_num = len(self.kg1.entities_set | self.kg2.entities_set)
        self.relations_num = len(self.kg1.relations_set | self.kg2.relations_set)
        self.attributes_num = len(self.kg1.attributes_set | self.kg2.attributes_set)


class ShareOntoKGs(KGs):
    def __init__(self, kg1: KG, kg2: KG, onto_kg: KG,
                 kg1_cv_links, kg2_cv_links, train_links, test_links,
                 kg1_seed_ent_types, kg2_seed_ent_types, training_data_folder,
                 valid_links=None, mode='mapping', onto_mode=None, ordered=True,
                 unsure_w=0, check_version=0, dataset_division='',
                 k=0, miss_completion=True, delay_weight=1.0,
                 name_dict=None):

        super().__init__(kg1, kg2, train_links, test_links, valid_links, mode, ordered)
        self.dataset_division = dataset_division
        self.training_data_folder = training_data_folder
        self.name_dict = name_dict

        # 本体信息
        onto_ent_ids = generate_mapping_id_oneKG(onto_kg.relation_triples_set, onto_kg.entities_set, ordered=ordered)
        onto_rel_ids = generate_mapping_id_oneKG(onto_kg.relation_triples_set, onto_kg.relations_set, ordered=ordered)
        onto_attr_ids = generate_mapping_id_oneKG(onto_kg.attribute_triples_set, onto_kg.attributes_set, ordered=ordered)

        self.onto_class_id_dict = onto_ent_ids
        self.onto_metarel_id_dict = onto_rel_ids

        # 利用训练种子生成Type矩阵
        onto_mat, kg1_seed_ent_types, kg2_seed_ent_types = self._load_supervised_type_mat(
            onto_ent_ids, train_links, kg1_seed_ent_types, kg2_seed_ent_types,
            unsure_w=unsure_w, check_version=check_version,
            k=k, miss_completion=miss_completion)
        self.onto_mat = onto_mat
        self.kg1_seed_ent_types = kg1_seed_ent_types
        self.kg2_seed_ent_types = kg2_seed_ent_types

        # 校正cv link中种子class和seed_ent_types不一致的情况
        revise_num = 0
        new_kg1_cv_links, new_kg2_cv_links = [], []
        for ent, ent_type in kg1_cv_links:
            if ent in kg1_seed_ent_types and kg1_seed_ent_types[ent] != ent_type:
                revise_num += 1
                new_kg1_cv_links.append((ent, kg1_seed_ent_types[ent]))
            else:
                new_kg1_cv_links.append((ent, ent_type))
        for ent, ent_type in kg2_cv_links:
            if ent in kg2_seed_ent_types and kg2_seed_ent_types[ent] != ent_type:
                revise_num += 1
                new_kg2_cv_links.append((ent, kg2_seed_ent_types[ent]))
            else:
                new_kg2_cv_links.append((ent, ent_type))
        kg1_cv_links = new_kg1_cv_links
        kg2_cv_links = new_kg2_cv_links
        print('revise num : ', revise_num)
        # raise Exception('Stop')

        # 生成class path字典
        self.uri_class_path = load_class_path(os.path.join(
            self.training_data_folder, self.dataset_division, 'class_path.json'))
        self.class_path = dict()
        for key, value in self.uri_class_path.items():
            self.class_path[onto_ent_ids[key]] = [onto_ent_ids[v] for v in value]
        # 将所有路径填补至等长
        self.class_max_depth, self.class_path_matrix, self.mask_path_matrix = self._padding_class_path(
            self.class_path, delay_weight=delay_weight)

        # 生成ent2onto字典
        self.url_ent2onto_dict = dict(self.kg1_seed_ent_types, **self.kg2_seed_ent_types)
        # 生成id编码的字典
        self.ent2onto_dict = dict()
        for key, value in self.url_ent2onto_dict.items():
            assert value in onto_ent_ids
            if key in self.kg1.entities_id_dict:
                self.ent2onto_dict[self.kg1.entities_id_dict[key]] = onto_ent_ids[value]
            elif key in self.kg2.entities_id_dict:
                self.ent2onto_dict[self.kg2.entities_id_dict[key]] = onto_ent_ids[value]
            else:
                assert 1 == 2

        # 生成本体到实体的字典，用于采样
        self.onto2ent_dict1 = dict()
        self.onto2ent_dict2 = dict()
        for ent, onto in self.ent2onto_dict.items():
            if ent in self.kg1.entities_id_dict.values():
                if onto not in self.onto2ent_dict1:
                    self.onto2ent_dict1[onto] = [ent]
                else:
                    self.onto2ent_dict1[onto].append(ent)
            elif ent in self.kg2.entities_id_dict.values():
                if onto not in self.onto2ent_dict2:
                    self.onto2ent_dict2[onto] = [ent]
                else:
                    self.onto2ent_dict2[onto].append(ent)
            else:
                raise Exception("error")

        self.train_ontologies1 = [self.ent2onto_dict[id] for id in self.train_entities1]
        self.train_ontologies2 = [self.ent2onto_dict[id] for id in self.train_entities2]
        self.test_ontologies1 = [self.ent2onto_dict[id] for id in self.test_entities1]
        self.test_ontologies2 = [self.ent2onto_dict[id] for id in self.test_entities2]

        # 验证集不为空
        if valid_links is not None:
            self.valid_ontologies1 = [self.ent2onto_dict[id] for id in self.valid_entities1]
            self.valid_ontologies2 = [self.ent2onto_dict[id] for id in self.valid_entities2]

        # 关系三元组到id的转换
        id_onto_relation_triples = uris_relation_triple_2ids(
            onto_kg.relation_triples_set, onto_ent_ids, onto_rel_ids)

        # 属性三元组到id的转换
        id_onto_attribute_triples = uris_attribute_triple_2ids(
            onto_kg.attribute_triples_set, onto_ent_ids, onto_attr_ids)

        self.uri_onto_kg = onto_kg

        onto_kg = KG(id_onto_relation_triples, id_onto_attribute_triples)
        onto_kg.set_id_dict(onto_ent_ids, onto_rel_ids, onto_attr_ids)

        # 两种语言的cv links合在一起处理
        cv_links = kg1_cv_links + kg2_cv_links
        self.uri_kg1_cv_links = kg1_cv_links
        self.uri_kg2_cv_links = kg2_cv_links
        self.uri_cv_links = cv_links
        self.all_kg_ent_id_dict = dict(self.kg1.entities_id_dict, **self.kg2.entities_id_dict)  # 合并两个实体KG的ent_id字典
        # print(len(self.kg1.entities_id_dict))
        # print(len(self.kg2.entities_id_dict))
        # print(len(self.all_kg_ent_id_dict))
        # if 'http://ja.dbpedia.org/resource/西ドイツ' not in self.all_kg_ent_id_dict:
        #     raise Exception('wrong')
        # print(list(self.all_kg_ent_id_dict.keys()))
        self.kg1_cv_links = uris_pair_2ids(self.uri_kg1_cv_links, self.all_kg_ent_id_dict, onto_ent_ids)
        self.kg2_cv_links = uris_pair_2ids(self.uri_kg2_cv_links, self.all_kg_ent_id_dict, onto_ent_ids)
        self.cv_links = uris_pair_2ids(self.uri_cv_links, self.all_kg_ent_id_dict, onto_ent_ids)
        # print('uri_kg1_cv_links : ',len(self.uri_kg1_cv_links))
        # print('uri_kg2_cv_links : ', len(self.uri_kg2_cv_links))
        # print('uri_cv_links : ', len(self.uri_cv_links))
        # print('kg1_cv_links : ', len(self.kg1_cv_links))
        # print('kg2_cv_links : ', len(self.kg2_cv_links))
        # print('cv_links : ', len(self.cv_links))
        # raise Exception('stop')

        self.instance_entities = [link[0] for link in self.cv_links]
        self.ontology_entities = [link[1] for link in self.cv_links]

        if onto_mode == 'swapping':   # 替换cv link的头实体，增强cv link个数
            sup_cv_links = generate_sup_cv_links(self.train_links, self.cv_links)
            self.cv_links.append(sup_cv_links)

        self.onto_kg = onto_kg

        self.onto_entities_num = len(self.onto_kg.entities_set)
        self.onto_relations_num = len(self.onto_kg.relations_set)
        self.onto_attributes_num = len(self.onto_kg.attributes_set)

    def _load_supervised_type_mat(self, onto_ent_ids, seed_links,
                                  kg1_seed_ent_types, kg2_seed_ent_types,
                                  unsure_w=0, check_version=0, filename='onto_check_mat.h5',
                                  k=0, miss_completion=False):
        type_relation = load_type_relation(os.path.join(
            self.training_data_folder, self.dataset_division), bracket=True)
        # 生成本体矩阵
        check_type_coherence(type_relation,
                             os.path.join(self.training_data_folder, self.dataset_division),
                             unsure_w=unsure_w, version=check_version,
                             save_name=filename, k=k)
        onto_check_mat = rd.load_onto_check_mat(os.path.join(
            self.training_data_folder, self.dataset_division, filename))
        onto_mat = onto_check_mat['onto_mat']
        onto2id_dict = onto_check_mat['onto2id_dict']

        # 按照onto_ent_ids重新排列onto_mat
        onto_id_ents = dict(zip(onto_ent_ids.values(), onto_ent_ids.keys()))  # 键值对互换
        # onto_id_ents = sorted(onto_id_ents.items(),key=lambda x:x[0],reverse=False)  # key从小到大排序
        new_onto_mat = np.zeros(shape=onto_mat.shape)
        for i in range(new_onto_mat.shape[0]):
            for j in range(new_onto_mat.shape[1]):
                type1 = onto_id_ents[i]
                type2 = onto_id_ents[j]
                new_onto_mat[i][j] = onto_mat[onto2id_dict[type1]][onto2id_dict[type2]]

        # 利用对齐种子校验mat
        total_uri_links = seed_links
        supervised_links_num = len(total_uri_links)
        print('supervised_links_num : ', supervised_links_num)
        miss_completion_num = 0
        for i in range(supervised_links_num):
            a, b = total_uri_links[i]
            atype, btype = kg1_seed_ent_types[a], kg2_seed_ent_types[b]
            # 处理缺失的情况
            if miss_completion:
                if atype == MISS_URI and btype != MISS_URI:
                    kg1_seed_ent_types[a] = btype
                    atype = btype
                    miss_completion_num += 1
                elif atype != MISS_URI and btype == MISS_URI:
                    kg2_seed_ent_types[b] = atype
                    btype = atype
                    miss_completion_num += 1
            # if 'http' not in atype:
            #     atype = 'http://dbpedia.org/ontology/' + atype
            # if 'http' not in btype:
            #     btype = 'http://dbpedia.org/ontology/' + btype
            # print(a, atype, b, btype)
            atype_id, btype_id = onto_ent_ids[atype], onto_ent_ids[btype]
            new_onto_mat[atype_id][btype_id] = 1

        print('new onto mat')
        print(new_onto_mat)
        print('miss completion num is {}'.format(miss_completion_num))

        return new_onto_mat, kg1_seed_ent_types, kg2_seed_ent_types

    def _padding_class_path(self, path_dict, delay_weight=1.0):

        def class_path_weight(x, delay_weight=1.0):
            w = 1
            for i in range(len(x)):
                if x[i] == 0:
                    x[i] = NEG_INF
                else:
                    x[i] = w
                    w *= delay_weight
            x = np.array(x)
            x = np.exp(x) / sum(np.exp(x))
            return list(x)

        max_depth = 0
        path_matrix = []
        mask_matrix = []
        for cls, path in path_dict.items():
            max_depth = max(max_depth, len(path))
        for cls, path in path_dict.items():
            cp = [0] * (max_depth - len(path)) + path
            mask = [0] * (max_depth - len(path)) + [1] * len(path)
            path_matrix.append(cp)
            mask_matrix.append(class_path_weight(mask, delay_weight=delay_weight))
        return max_depth, path_matrix, mask_matrix


def read_kgs_from_folder(training_data_folder, division, mode, ordered, lang=None, remove_unlinked=False,
                         onto_valid=True, onto_mode=None, seed_ent_type='normal',
                         unsure_w=0, check_version=0, data_version='raw', dataset_division='',
                         k=0, delay_weight=0, seed_ratio=0.):
    '''
    training_data_folder : 数据集目录，例如datasets/D_W_15K_V1
    division : fold的目录
    mode : 对齐方式, sharing, mapping, swapping ......
    ordered : uri编号的方式，True是按照频率编号
    返回两个KG组成的一个KG
    '''
    if 'raw' in training_data_folder.lower():
        # ontology-enhanced KG loader
        return read_kgs_from_dbp_raw(training_data_folder, lang, division, mode, ordered,
                                     remove_unlinked=remove_unlinked, onto_valid=onto_valid, seed_ent_type=seed_ent_type,
                                     onto_mode=onto_mode, unsure_w=unsure_w, check_version=check_version,
                                     dataset_division=dataset_division, data_version=data_version,
                                     k=k, delay_weight=delay_weight, seed_ratio=seed_ratio)
    if 'dbp15k' in training_data_folder.lower() or 'dwy100k' in training_data_folder.lower():
        return read_kgs_from_dbp_dwy(training_data_folder, division, mode, ordered, remove_unlinked=remove_unlinked)
    kg1_relation_triples, _, _ = read_relation_triples(training_data_folder + 'rel_triples_1', sep='\t')
    kg2_relation_triples, _, _ = read_relation_triples(training_data_folder + 'rel_triples_2', sep='\t')
    kg1_attribute_triples, _, _ = read_attribute_triples(training_data_folder + 'attr_triples_1')
    kg2_attribute_triples, _, _ = read_attribute_triples(training_data_folder + 'attr_triples_2')

    train_links = read_links(training_data_folder + division + 'train_links')
    valid_links = read_links(training_data_folder + division + 'valid_links')
    test_links = read_links(training_data_folder + division + 'test_links')

    if remove_unlinked:
        links = train_links + valid_links + test_links
        kg1_relation_triples = remove_unlinked_triples(kg1_relation_triples, links)
        kg2_relation_triples = remove_unlinked_triples(kg2_relation_triples, links)

    kg1 = KG(kg1_relation_triples, kg1_attribute_triples)
    kg2 = KG(kg2_relation_triples, kg2_attribute_triples)

    if onto_valid:
        # 本体文件存放在dbp15K_raw_data路径下
        onto_kg_relation_triples, _, _ = read_relation_triples(
            os.path.join(training_data_folder, 'onto_subClassOf_triples'), bracket=True)
        onto_kg_attribute_triples, _, _ = read_attribute_triples(
            os.path.join(training_data_folder, 'onto_attr_triples'), bracket=True)

        kg1_cv_links = read_links(os.path.join(training_data_folder, 'crossview_link_1'))
        kg2_cv_links = read_links(os.path.join(training_data_folder, 'crossview_link_2'))

        # 加载对齐种子的type文件，两种类型：normal 保留不一致的结果  unify:将对齐种子的type强行一致
        kg1_seed_ent_types = []
        kg2_seed_ent_types = []
        kg1_seed_ent_types = read_entType_file(
            os.path.join(training_data_folder, 'crossview_link_1'), sep='\t')
        kg2_seed_ent_types = read_entType_file(
            os.path.join(training_data_folder, 'crossview_link_2'), sep='\t')

        # for D-W datasets
        ent2name = load_name_dicts(os.path.join(training_data_folder, 'attr_triples_2'))

        onto_kg = KG(onto_kg_relation_triples, onto_kg_attribute_triples)
        kgs = ShareOntoKGs(kg1, kg2, onto_kg, kg1_cv_links, kg2_cv_links, train_links, test_links,
                           kg1_seed_ent_types, kg2_seed_ent_types, training_data_folder,
                           valid_links=valid_links, mode=mode, onto_mode=onto_mode,
                           ordered=ordered, unsure_w=unsure_w,
                           check_version=check_version, dataset_division=dataset_division,
                           k=k, delay_weight=delay_weight, name_dict=ent2name)
    else:
        kgs = KGs(kg1, kg2, train_links, test_links, valid_links=valid_links, mode=mode, ordered=ordered)
    return kgs


def read_kgs_from_dbp_raw(training_data_folder, lang, division, mode, ordered,
                          remove_unlinked=False, onto_valid=True, seed_ent_type='normal',
                          onto_mode=None, unsure_w=0, check_version=0, data_version='raw',
                          dataset_division='', k=0, seed_ratio=0.3, shuffle_seed=False, delay_weight=1.0):
    '''
    training_data_folder : 数据集目录，例如datasets/B_W_15K_V1
    lang : 对齐的语言
    division : fold的目录
    mode : 对齐方式, sharing, mapping, swapping ......
    ordered : uri编号的方式，True是按照频率编号
    返回两个KG组成的一个KG
    '''
    print('read kgs from dbpedia raw data !')
    kg1_lang, kg2_lang = lang.split('_')[-2:]
    if data_version == 'raw':
        kg1_relation_triples, _, _ = read_relation_triples(
            os.path.join(training_data_folder, kg1_lang+'_rel_triples'))
        kg2_relation_triples, _, _ = read_relation_triples(
            os.path.join(training_data_folder, kg2_lang+'_rel_triples'))
        kg1_attribute_triples, _, _ = read_attribute_triples(
            os.path.join(training_data_folder, kg1_lang+'_att_triples'), bracket=True)
        kg2_attribute_triples, _, _ = read_attribute_triples(
            os.path.join(training_data_folder, kg2_lang+'_att_triples'), bracket=True)
    elif data_version == 'popular':
        kg1_relation_triples, _, _ = read_relation_triples(
            os.path.join(training_data_folder, kg1_lang + '_popular_rel_triples'))
        kg2_relation_triples, _, _ = read_relation_triples(
            os.path.join(training_data_folder, kg2_lang + '_popular_rel_triples'))
        kg1_attribute_triples, _, _ = read_attribute_triples(
            os.path.join(training_data_folder, kg1_lang + '_att_triples'), bracket=True)
        kg2_attribute_triples, _, _ = read_attribute_triples(
            os.path.join(training_data_folder, kg2_lang + '_att_triples'), bracket=True)
        # print(len(kg1_attribute_triples),len(kg2_attribute_triples))
    else:
        raise Exception('unvaild data version.')

    if division:
        train_links = read_links(training_data_folder + division + 'train_links')
        valid_links = read_links(training_data_folder + division + 'valid_links')
        test_links = read_links(training_data_folder + division + 'test_links')
        if abs(seed_ratio) > 0:
            total_seed_num = len(train_links) + len(valid_links) + len(test_links)
            sup_seed_num = int(total_seed_num * seed_ratio)
            if len(train_links) >= sup_seed_num:
                train_links = train_links[:sup_seed_num]
            else:
                app_seed_num = sup_seed_num - len(train_links)
                train_links = train_links + test_links[:app_seed_num]
    else:
        all_links = read_links(os.path.join(training_data_folder, 'ent_ILLs'))
        if shuffle_seed:
            all_links = random.shuffle(all_links)
        train_nums = int(len(all_links)*seed_ratio)
        train_links = all_links[:train_nums]
        valid_links = all_links[train_nums:]
        test_links = all_links[train_nums:]

    print('train seed nums : {}'.format(len(train_links)))

    if remove_unlinked:
        links = train_links + valid_links + test_links
        kg1_relation_triples = remove_unlinked_triples(kg1_relation_triples, links)
        kg2_relation_triples = remove_unlinked_triples(kg2_relation_triples, links)

    kg1 = KG(kg1_relation_triples, kg1_attribute_triples)
    kg2 = KG(kg2_relation_triples, kg2_attribute_triples)

    if onto_valid:
        # 本体文件存放在dbp15K_raw_data路径下
        onto_kg_relation_triples, _, _ = read_relation_triples(
            os.path.join(training_data_folder, 'onto_subClassOf_triples'), bracket=True)
        onto_kg_attribute_triples, _, _ = read_attribute_triples(
            os.path.join(training_data_folder, 'new_onto_attr_triples'), bracket=True)

        kg1_cv_links = read_links(os.path.join(training_data_folder, kg1_lang+'_crossview_link'))
        kg2_cv_links = read_links(os.path.join(training_data_folder, kg2_lang+'_crossview_link'))

        # 加载对齐种子的type文件，两种类型：normal 保留不一致的结果  unify:将对齐种子的type强行一致
        kg1_seed_ent_types = []
        kg2_seed_ent_types = []
        if seed_ent_type == 'normal':
            kg1_seed_ent_types = read_entType_file(os.path.join(training_data_folder,
                                                                kg1_lang + '_align_types'))
            kg2_seed_ent_types = read_entType_file(os.path.join(training_data_folder,
                                                                kg2_lang + '_align_types'))
        elif seed_ent_type == 'unify':
            kg1_seed_ent_types = read_entType_file(os.path.join(training_data_folder,
                                                                kg1_lang + '_unify_align_types'))
            kg2_seed_ent_types = read_entType_file(os.path.join(training_data_folder,
                                                                kg2_lang + '_unify_align_types'))
        elif seed_ent_type == 'normal_c':
            kg1_seed_ent_types = read_entType_file(os.path.join(training_data_folder,
                                                                kg1_lang + '_align_completion_types'))
            kg2_seed_ent_types = read_entType_file(os.path.join(training_data_folder,
                                                                kg2_lang + '_align_completion_types'))
        elif seed_ent_type == 'normal_cd':
            kg1_seed_ent_types = read_entType_file(os.path.join(training_data_folder,
                                                                kg1_lang + '_align_completion_deep4_types'))
            kg2_seed_ent_types = read_entType_file(os.path.join(training_data_folder,
                                                                kg2_lang + '_align_completion_deep4_types'))
        elif seed_ent_type == 'src_c':
            kg1_seed_ent_types = read_entType_file(os.path.join(training_data_folder,
                                                                kg1_lang + '_completion_types'))
            kg2_seed_ent_types = read_entType_file(os.path.join(training_data_folder,
                                                                kg2_lang + '_completion_types'))
        elif seed_ent_type == 'src_cd':
            kg1_seed_ent_types = read_entType_file(os.path.join(training_data_folder,
                                                                kg1_lang + '_completion_deep4_types'))
            kg2_seed_ent_types = read_entType_file(os.path.join(training_data_folder,
                                                                kg2_lang + '_completion_deep4_types'))
        elif seed_ent_type == 'pesudo_c':
            kg1_seed_ent_types = read_entType_file(os.path.join(training_data_folder,
                                                                kg1_lang + '_pesudo_completion_types'))
            kg2_seed_ent_types = read_entType_file(os.path.join(training_data_folder,
                                                                kg2_lang + '_pesudo_completion_types'))
        elif seed_ent_type == 'pesudo_cd':
            kg1_seed_ent_types = read_entType_file(os.path.join(training_data_folder,
                                                                kg1_lang + '_pesudo_completion_deep4_types'))
            kg2_seed_ent_types = read_entType_file(os.path.join(training_data_folder,
                                                                kg2_lang + '_pesudo_completion_deep4_types'))
        elif seed_ent_type == 'zh_en':
            percent = 100
            kg1_seed_ent_types = read_entType_file(os.path.join(training_data_folder,
                                                                kg1_lang + '_' + str(percent) + 'percent_completion_types'))
            kg2_seed_ent_types = read_entType_file(os.path.join(training_data_folder,
                                                                kg2_lang + '_' + str(percent) + 'percent_completion_types'))

        # cv_links = kg1_cv_links + kg2_cv_links  # 合并两个cv link
        onto_kg = KG(onto_kg_relation_triples, onto_kg_attribute_triples)
        kgs = ShareOntoKGs(kg1, kg2, onto_kg, kg1_cv_links, kg2_cv_links, train_links, test_links,
                           kg1_seed_ent_types, kg2_seed_ent_types, training_data_folder,
                           valid_links=valid_links, mode=mode, onto_mode=onto_mode,
                           ordered=ordered, unsure_w=unsure_w,
                           check_version=check_version, dataset_division=dataset_division,
                           k=k, delay_weight=delay_weight)
    else:
        kgs = KGs(kg1, kg2, train_links, test_links, valid_links=valid_links, mode=mode, ordered=ordered)
    return kgs


def read_reversed_kgs_from_folder(training_data_folder, division, mode, ordered, remove_unlinked=False):
    kg1_relation_triples, _, _ = read_relation_triples(training_data_folder + 'rel_triples_2')
    kg2_relation_triples, _, _ = read_relation_triples(training_data_folder + 'rel_triples_1')
    kg1_attribute_triples, _, _ = read_attribute_triples(training_data_folder + 'attr_triples_2')
    kg2_attribute_triples, _, _ = read_attribute_triples(training_data_folder + 'attr_triples_1')

    temp_train_links = read_links(training_data_folder + division + 'train_links')
    temp_valid_links = read_links(training_data_folder + division + 'valid_links')
    temp_test_links = read_links(training_data_folder + division + 'test_links')
    train_links = [(j, i) for i, j in temp_train_links]
    valid_links = [(j, i) for i, j in temp_valid_links]
    test_links = [(j, i) for i, j in temp_test_links]

    if remove_unlinked:
        links = train_links + valid_links + test_links
        kg1_relation_triples = remove_unlinked_triples(kg1_relation_triples, links)
        kg2_relation_triples = remove_unlinked_triples(kg2_relation_triples, links)

    kg1 = KG(kg1_relation_triples, kg1_attribute_triples)
    kg2 = KG(kg2_relation_triples, kg2_attribute_triples)
    kgs = KGs(kg1, kg2, train_links, test_links, valid_links=valid_links, mode=mode, ordered=ordered)
    return kgs


def read_kgs_from_files(kg1_relation_triples, kg2_relation_triples,
                        kg1_attribute_triples, kg2_attribute_triples,
                        train_links, valid_links, test_links, mode):
    kg1 = KG(kg1_relation_triples, kg1_attribute_triples)
    kg2 = KG(kg2_relation_triples, kg2_attribute_triples)
    kgs = KGs(kg1, kg2, train_links, test_links, valid_links=valid_links, mode=mode)
    return kgs


def read_kgs_from_dbp_dwy(folder, division, mode, ordered, remove_unlinked=False):
    folder = folder + division
    kg1_relation_triples, _, _ = read_relation_triples(folder + 'triples_1')
    kg2_relation_triples, _, _ = read_relation_triples(folder + 'triples_2')
    if os.path.exists(folder + 'sup_pairs'):
        train_links = read_links(folder + 'sup_pairs')
    else:
        train_links = read_links(folder + 'sup_ent_ids')
    if os.path.exists(folder + 'ref_pairs'):
        test_links = read_links(folder + 'ref_pairs')
    else:
        test_links = read_links(folder + 'ref_ent_ids')
    print()
    if remove_unlinked:
        for i in range(10000):
            print("removing times:", i)
            links = train_links + test_links
            kg1_relation_triples = remove_unlinked_triples(kg1_relation_triples, links)
            kg2_relation_triples = remove_unlinked_triples(kg2_relation_triples, links)
            n1 = len(kg1_relation_triples)
            n2 = len(kg2_relation_triples)
            train_links, test_links = remove_no_triples_link(kg1_relation_triples, kg2_relation_triples,
                                                             train_links, test_links)
            links = train_links + test_links
            kg1_relation_triples = remove_unlinked_triples(kg1_relation_triples, links)
            kg2_relation_triples = remove_unlinked_triples(kg2_relation_triples, links)
            n11 = len(kg1_relation_triples)
            n22 = len(kg2_relation_triples)
            if n1 == n11 and n2 == n22:
                break
            print()

    kg1 = KG(kg1_relation_triples, list())
    kg2 = KG(kg2_relation_triples, list())
    kgs = KGs(kg1, kg2, train_links, test_links, mode=mode, ordered=ordered)
    return kgs


def remove_no_triples_link(kg1_relation_triples, kg2_relation_triples, train_links, test_links):
    kg1_entities, kg2_entities = set(), set()
    for h, r, t in kg1_relation_triples:
        kg1_entities.add(h)
        kg1_entities.add(t)
    for h, r, t in kg2_relation_triples:
        kg2_entities.add(h)
        kg2_entities.add(t)
    print("before removing links with no triples:", len(train_links), len(test_links))
    new_train_links, new_test_links = set(), set()
    for i, j in train_links:
        if i in kg1_entities and j in kg2_entities:
            new_train_links.add((i, j))
    for i, j in test_links:
        if i in kg1_entities and j in kg2_entities:
            new_test_links.add((i, j))
    print("after removing links with no triples:", len(new_train_links), len(new_test_links))
    return list(new_train_links), list(new_test_links)


def remove_unlinked_triples(triples, links):
    print("before removing unlinked triples:", len(triples))
    linked_entities = set()
    for i, j in links:
        linked_entities.add(i)
        linked_entities.add(j)
    linked_triples = set()
    for h, r, t in triples:
        if h in linked_entities and t in linked_entities:
            linked_triples.add((h, r, t))
    print("after removing unlinked triples:", len(linked_triples))
    return linked_triples
