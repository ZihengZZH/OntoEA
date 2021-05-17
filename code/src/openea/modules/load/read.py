import os
import h5py
import json
import numpy as np


def load_embeddings(file_name):
    if os.path.exists(file_name):
        return np.load(file_name)
    return None


# 按照实体出现的频率排序
def sort_elements(triples, elements_set):
    dic = dict()
    for s, p, o in triples:
        if s in elements_set:
            dic[s] = dic.get(s, 0) + 1
        if p in elements_set:
            dic[p] = dic.get(p, 0) + 1
        if o in elements_set:
            dic[o] = dic.get(o, 0) + 1
    # firstly sort by values (i.e., frequencies), if equal, by keys (i.e, URIs)
    sorted_list = sorted(dic.items(), key=lambda x: (x[1], x[0]), reverse=True)
    ordered_elements = [x[0] for x in sorted_list]
    return ordered_elements, dic


def generate_sharing_id(train_links, kg1_triples, kg1_elements, kg2_triples, kg2_elements, ordered=True):
    '''
    sharing的意思是两个对齐的实体的编号id是相同的
    '''
    ids1, ids2 = dict(), dict()
    if ordered:
        linked_dic = dict()
        for x, y in train_links:
            linked_dic[y] = x
        kg2_linked_elements = [x[1] for x in train_links]
        kg2_unlinked_elements = set(kg2_elements) - set(kg2_linked_elements)
        ids1, ids2 = generate_mapping_id(kg1_triples, kg1_elements, kg2_triples, kg2_unlinked_elements, ordered=ordered)
        for ele in kg2_linked_elements:   # 将两个对齐的实体编码为同一个id
            ids2[ele] = ids1[linked_dic[ele]]
    else:    # 为每个元素按照列表顺序编号
        index = 0
        for e1, e2 in train_links:
            assert e1 in kg1_elements
            assert e2 in kg2_elements
            ids1[e1] = index
            ids2[e2] = index
            index += 1
        for ele in kg1_elements:
            if ele not in ids1:
                ids1[ele] = index
                index += 1
        for ele in kg2_elements:
            if ele not in ids2:
                ids2[ele] = index
                index += 1
    assert len(ids1) == len(set(kg1_elements))
    assert len(ids2) == len(set(kg2_elements))
    return ids1, ids2


def generate_mapping_id(kg1_triples, kg1_elements, kg2_triples, kg2_elements, ordered=True):
    '''
    ordered : True 每个KG按照实体出现的频率排序，然后从高到低编号，注意所有的实体的一起编号的，也就是72~73 line所示
    ordered : False 每个KG按照列表的顺序编号
    '''
    ids1, ids2 = dict(), dict()
    if ordered:
        kg1_ordered_elements, _ = sort_elements(kg1_triples, kg1_elements)    # 按照实体出现的频率排序
        kg2_ordered_elements, _ = sort_elements(kg2_triples, kg2_elements)
        n1 = len(kg1_ordered_elements)
        n2 = len(kg2_ordered_elements)
        n = max(n1, n2)
        for i in range(n):
            if i < n1 and i < n2:
                ids1[kg1_ordered_elements[i]] = i * 2
                ids2[kg2_ordered_elements[i]] = i * 2 + 1
            elif i >= n1:
                ids2[kg2_ordered_elements[i]] = n1 * 2 + (i - n1)
            else:
                ids1[kg1_ordered_elements[i]] = n2 * 2 + (i - n2)
    else:
        index = 0
        for ele in kg1_elements:
            if ele not in ids1:
                ids1[ele] = index
                index += 1
        for ele in kg2_elements:
            if ele not in ids2:
                ids2[ele] = index
                index += 1
    assert len(ids1) == len(set(kg1_elements))
    assert len(ids2) == len(set(kg2_elements))
    return ids1, ids2


def generate_mapping_id_oneKG(kg1_triples, kg1_elements, ordered=True):
    '''
    ordered : True 每个KG按照实体出现的频率排序，然后从高到低编号，注意所有的实体的一起编号的，也就是72~73 line所示
    ordered : False 每个KG按照列表的顺序编号
    '''
    ids1 = dict()
    if ordered:
        kg1_ordered_elements, _ = sort_elements(kg1_triples, kg1_elements)    # 按照实体出现的频率排序
        n1 = len(kg1_ordered_elements)
        for i in range(n1):
            ids1[kg1_ordered_elements[i]] = i
    else:
        index = 0
        for ele in kg1_elements:
            if ele not in ids1:
                ids1[ele] = index
                index += 1
    assert len(ids1) == len(set(kg1_elements))
    return ids1


def uris_list_2ids(uris, ids):
    id_uris = list()
    for u in uris:
        assert u in ids
        id_uris.append(ids[u])
    assert len(id_uris) == len(set(uris))
    return id_uris


def uris_pair_2ids(uris, ids1, ids2):
    id_uris = list()
    for u1, u2 in uris:
        # print(u1,u2)
        # assert u1 in ids1
        # assert u2 in ids2
        if u1 in ids1 and u2 in ids2:
            id_uris.append((ids1[u1], ids2[u2]))
    # assert len(id_uris) == len(set(uris))
    return id_uris


def uris_relation_triple_2ids(uris, ent_ids, rel_ids):
    id_uris = list()
    for u1, u2, u3 in uris:
        assert u1 in ent_ids
        assert u2 in rel_ids
        assert u3 in ent_ids
        id_uris.append((ent_ids[u1], rel_ids[u2], ent_ids[u3]))
    assert len(id_uris) == len(set(uris))
    return id_uris


def uris_attribute_triple_2ids(uris, ent_ids, attr_ids, debug=False):
    id_uris = list()
    for u1, u2, u3 in uris:
        if debug:
            assert u1 in ent_ids
            assert u2 in attr_ids
        else:
            if u1 not in ent_ids:
                continue
        id_uris.append((ent_ids[u1], attr_ids[u2], u3))
    # assert len(id_uris) == len(set(uris))
    return id_uris


def generate_sup_relation_triples_one_link(e1, e2, rt_dict, hr_dict):
    new_triples = set()
    for r, t in rt_dict.get(e1, set()):
        new_triples.add((e2, r, t))
    for h, r in hr_dict.get(e1, set()):
        new_triples.add((h, r, e2))
    return new_triples


def generate_sup_relation_triples(sup_links, rt_dict1, hr_dict1, rt_dict2, hr_dict2):
    new_triples1, new_triples2 = set(), set()
    for ent1, ent2 in sup_links:
        new_triples1 |= (generate_sup_relation_triples_one_link(ent1, ent2, rt_dict1, hr_dict1))
        new_triples2 |= (generate_sup_relation_triples_one_link(ent2, ent1, rt_dict2, hr_dict2))
    print("supervised relation triples: {}, {}".format(len(new_triples1), len(new_triples2)))
    return new_triples1, new_triples2


def generate_sup_cv_links(algin_links, cv_links):
    new_cv_links = set()
    cv_links_dict = dict(cv_links)
    for ent1, ent2 in algin_links:
        onto1 = cv_links_dict.get(ent1, -1)
        onto2 = cv_links_dict.get(ent2, -1)
        if onto1 == onto2:
            continue
        elif onto1 == -1:
            new_cv_links.add((ent1, onto2))
        elif onto2 == -1:
            new_cv_links.add((ent2, onto1))
        else:
            new_cv_links.add((ent1, onto2))
            new_cv_links.add((ent2, onto1))
    print("supervised cross-KG links : {}".format(len(new_cv_links)))
    return new_cv_links


def generate_sup_attribute_triples_one_link(e1, e2, av_dict):
    new_triples = set()
    for a, v in av_dict.get(e1, set()):
        new_triples.add((e2, a, v))
    return new_triples


def generate_sup_attribute_triples(sup_links, av_dict1, av_dict2):
    new_triples1, new_triples2 = set(), set()
    for ent1, ent2 in sup_links:
        new_triples1 |= (generate_sup_attribute_triples_one_link(ent1, ent2, av_dict1))
        new_triples2 |= (generate_sup_attribute_triples_one_link(ent2, ent1, av_dict2))
    print("supervised attribute triples: {}, {}".format(len(new_triples1), len(new_triples2)))
    return new_triples1, new_triples2


def read_relation_triples(file_path, bracket=False, sep=' '):
    print("read relation triples:", file_path)
    if file_path is None:
        return set(), set(), set()
    triples = set()
    entities, relations = set(), set()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip('\n').split(sep)
        assert len(params) == 3 or len(params) == 4
        h = params[0].strip() if not bracket else params[0].strip()[1:-1]
        r = params[1].strip() if not bracket else params[1].strip()[1:-1]
        t = params[2].strip() if not bracket else params[2].strip()[1:-1]
        triples.add((h, r, t))
        entities.add(h)
        entities.add(t)
        relations.add(r)
    return triples, entities, relations


def read_links(file_path):
    if not os.path.exists(file_path):
        print("{} is not found.".format(file_path))
        return list()
    print("read links:", file_path)
    links = list()
    refs = list()
    reft = list()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 2
        e1 = params[0].strip()
        e2 = params[1].strip()
        refs.append(e1)
        reft.append(e2)
        links.append((e1, e2))
    assert len(refs) == len(reft)
    return links


# 加载跨语言对齐文件
def read_ent_ills(file_path):
    sList, tList = [], []   # 分别保存s,t的实体
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            s, t = line.strip().split()   # 对齐实体(s,t)
            sList.append(s)
            tList.append(t)
    # print("line nums : {} , ent nums : {}".format(len(lines),len(sList)))
    return sList, tList


def read_dict(file_path):
    if not os.path.exists(file_path):
        return None
    file = open(file_path, 'r', encoding='utf8')
    ids = dict()
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 2
        ids[params[0]] = int(params[1])
    file.close()
    return ids


def read_pair_ids(file_path):
    if not os.path.exists(file_path):
        return None
    file = open(file_path, 'r', encoding='utf8')
    pairs = list()
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 2
        pairs.append((int(params[0]), int(params[1])))
    file.close()
    return pairs


def pair2file(file, pairs):
    if pairs is None:
        return
    with open(file, 'w', encoding='utf8') as f:
        for i, j in pairs:
            f.write(str(i) + '\t' + str(j) + '\n')
        f.close()


def dict2file(file, dic):
    if dic is None:
        return
    with open(file, 'w', encoding='utf8') as f:
        for i, j in dic.items():
            f.write(str(i) + '\t' + str(j) + '\n')
        f.close()
    print(file, "saved.")


def line2file(file, lines):
    if lines is None:
        return
    with open(file, 'w', encoding='utf8') as f:
        for line in lines:
            f.write(line + '\n')
        f.close()
    print(file, "saved.")


def radio_2file(radio, folder):
    path = folder + str(radio).replace('.', '_')
    if not os.path.exists(path):
        os.makedirs(path)
    return path + '/'


def save_results(folder, rest_12, rest_12_csls=None):
    if not os.path.exists(folder):
        os.makedirs(folder)
    pair2file(folder + 'alignment_results_12', rest_12)
    if rest_12_csls is not None:
        pair2file(folder + 'alignment_results_12_CSLS', rest_12_csls)
    print("Results saved!")


def save_embeddings(folder, kgs, ent_embeds, rel_embeds, attr_embeds, mapping_mat=None, rev_mapping_mat=None):
    if not os.path.exists(folder):
        os.makedirs(folder)
    if ent_embeds is not None:
        np.save(folder + 'ent_embeds.npy', ent_embeds)
    if rel_embeds is not None:
        np.save(folder + 'rel_embeds.npy', rel_embeds)
    if attr_embeds is not None:
        np.save(folder + 'attr_embeds.npy', attr_embeds)
    if mapping_mat is not None:
        np.save(folder + 'mapping_mat.npy', mapping_mat)
    if rev_mapping_mat is not None:
        np.save(folder + 'rev_mapping_mat.npy', rev_mapping_mat)
    dict2file(folder + 'kg1_ent_ids', kgs.kg1.entities_id_dict)
    dict2file(folder + 'kg2_ent_ids', kgs.kg2.entities_id_dict)
    dict2file(folder + 'kg1_rel_ids', kgs.kg1.relations_id_dict)
    dict2file(folder + 'kg2_rel_ids', kgs.kg2.relations_id_dict)
    dict2file(folder + 'kg1_attr_ids', kgs.kg1.attributes_id_dict)
    dict2file(folder + 'kg2_attr_ids', kgs.kg2.attributes_id_dict)
    print("Embeddings saved!")


def save_onto_embeddings(folder, kgs, ent_embeds, rel_embeds, attr_embeds, mapping_mat=None, rev_mapping_mat=None):
    if not os.path.exists(folder):
        os.makedirs(folder)
    if ent_embeds is not None:
        np.save(folder + 'onto_ent_embeds.npy', ent_embeds)
    if rel_embeds is not None:
        np.save(folder + 'onto_rel_embeds.npy', rel_embeds)
    if attr_embeds is not None:
        np.save(folder + 'onto_attr_embeds.npy', attr_embeds)
    if mapping_mat is not None:
        np.save(folder + 'onto_mapping_mat.npy', mapping_mat)
    if rev_mapping_mat is not None:
        np.save(folder + 'onto_rev_mapping_mat.npy', rev_mapping_mat)
    dict2file(folder + 'onto_kg_ent_ids', kgs.onto_kg.entities_id_dict)
    dict2file(folder + 'onto_kg_rel_ids', kgs.onto_kg.relations_id_dict)
    dict2file(folder + 'onto_kg_attr_ids', kgs.onto_kg.attributes_id_dict)
    pair2file(folder + 'cv_link_ids', kgs.cv_links)
    print("Embeddings saved!")


def filter_bracket(s):
    if s.strip().startswith('<') and s.strip().endswith('>'):
        return s.strip()[1:-1]
    else:
        return s.strip()


def read_attribute_triples(file_path, bracket=False):
    '''
    bracket : 是否移除<>
    '''
    print("read attribute triples:", file_path)
    if file_path is None:
        return set(), set(), set()
    if file_path is None:
        return set(), set(), set()
    if not os.path.exists(file_path):
        return set(), set(), set()
    triples = set()
    entities, attributes = set(), set()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip().strip('\n').split()   # line.strip().strip('\n').split('\t')
        if len(params) < 3:
            continue
        # print(params)
        head = filter_bracket(params[0])
        attr = filter_bracket(params[1])
        value = filter_bracket(params[2])

        if len(params) > 3:
            for p in params[3:]:
                if p.strip() != '.':   # 过滤掉以 . 结尾的三元组的最后一个字符
                    value = value + ' ' + p.strip()
        value = value.strip().rstrip('.').strip()
        entities.add(head)
        attributes.add(attr)
        triples.add((head, attr, value))
    return triples, entities, attributes


def read_entType_file(path, sep=' '):
    if not os.path.exists(path):
        print("path {} is not found.".format(path))
        return {}
    ent_type_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            items = line.strip().split(sep)
            assert len(items) == 2
            ent_type_dict[items[0]] = items[1]
    print('load {} success'.format(path))
    return ent_type_dict


def read_onto_file(path, blacket=True):
    onto_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            item = line.strip().split()
            assert len(item) >= 3
            if blacket:
                item = [i[1:-1] for i in item]
            onto_dict[item[0]] = item[2]
    return onto_dict


def load_onto_check_mat(path):
    file = h5py.File(path, 'r')
    onto_mat = file['onto_mat'][:]   # h5py datasets转numpy
    onto_name = [n.decode('utf-8') for n in file['onto_name'][()]]
    onto_id = file['onto_id'][()]
    onto2id_dict = dict(zip(onto_name, onto_id))
    file.close()
    # print(onto_mat.shape)
    # print(list(onto2id_dict.keys())[:2])
    return {
        'onto_mat': onto_mat,
        'onto2id_dict': onto2id_dict
    }


def load_class_path(path):
    with open(path, 'r', encoding='utf-8') as f:
        class_path = json.load(f)
    return class_path


def load_name_dicts(path):
    ALTER = [
        'http://www.wikidata.org/entity/P1476',
        'http://www.wikidata.org/entity/P373'
    ]
    with open(path, 'r', encoding='utf-8') as f_in:
        lines = [line.strip().split('\t') for line in f_in]
    ent2name = {}
    count = 0
    for line in lines:
        if line[1] in ALTER:
            ent2name[line[0]] = line[2]
            count += 1
    print("# alternative labels == %d out of %d" % (count, len(lines)))
    return ent2name
