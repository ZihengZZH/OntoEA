
import os
import math
from openea.modules.utils import check
from openea.modules.load import read


'''
需要读取的数据：
lang1 : KG文件，cv link文件，ent class文件
根据关系统计头尾实体的class的频率，利用阈值或者熵的方式选取每种关系最适合的class
统计选取的class粒度，避免太粗
统计两个对齐的实体class一致的比例
'''


THING_URI = "http://www.w3.org/2002/07/owl#Thing"
check_func = check.check_type_v2


############################################################
#########################  load file #######################
############################################################


def load_crossview_link(path):
    ent2onto_dict = dict()
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        ent, onto = line.split()
        if ent not in ent2onto_dict:
            ent2onto_dict[ent] = onto
        else:
            raise Exception('one to many ontologies : {}'.format(ent))
    return ent2onto_dict


def load_triples_dict(path):
    h_rt, r_ht, t_hr = {}, {}, {}
    triples = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        items = line.split()
        assert len(items) == 3
        h, r, t = items
        triples.append((h, r, t))
        if h not in h_rt:
            h_rt[h] = [(r, t)]
        else:
            h_rt[h].append((r, t))
        if r not in r_ht:
            r_ht[r] = [(h, t)]
        else:
            r_ht[r].append((h, t))
        if t not in t_hr:
            t_hr[t] = [(h, r)]
        else:
            t_hr[t].append((h, r))
    return {
        "h_rt": h_rt,
        "r_ht": r_ht,
        "t_hr": t_hr,
        "triples": triples
    }


def load_datasets_info(folder, lang, unify):
    print('start load data ...')
    lang1, lang2 = lang.split('_')
    ent_ills = read.read_ent_ills(os.path.join(folder, lang, 'ent_ILLs'))   # 返回一个列表，每个元素是一个元组
    # lang1_cv_link = load_crossview_link(os.path.join(path,lang1+'_crossview_link'))
    # lang2_cv_link = load_crossview_link(os.path.join(path, lang2+'_crossview_link'))
    if unify == 0:
        lang1_cv_link = load_crossview_link(os.path.join(folder, lang, lang1 + '_unify_align_types'))
        lang2_cv_link = load_crossview_link(os.path.join(folder, lang, lang2 + '_unify_align_types'))
    elif unify == 1:
        lang1_cv_link = load_crossview_link(os.path.join(folder, lang, lang1 + '_types'))
        lang2_cv_link = load_crossview_link(os.path.join(folder, lang, lang2 + '_types'))
    elif unify == 2:
        lang1_cv_link = load_crossview_link(os.path.join(folder, lang, lang1 + '_align_types'))
        lang2_cv_link = load_crossview_link(os.path.join(folder, lang, lang2 + '_align_types'))

    lang1_kg = load_triples_dict(os.path.join(folder, lang, lang1 + "_popular_rel_triples"))
    lang2_kg = load_triples_dict(os.path.join(folder, lang, lang2 + "_popular_rel_triples"))
    onto_dict = check.load_type_relation(folder)
    print('load data finish ... ')
    return {
        "ent_ills": ent_ills,
        "lang1_cv_link": lang1_cv_link,
        "lang2_cv_link": lang2_cv_link,
        "lang1_kg": lang1_kg,
        "lang2_kg": lang2_kg,
        "onto_dict": onto_dict
    }


##################################################################
######################### ontology info ##########################
##################################################################


def print_ontology_structure(onto_dict, cv_link1, cv_link2):
    tree_dict = {}
    cv_link = dict(cv_link1, **cv_link2)
    print('cv link nums : {}'.format(len(cv_link)))
    for ent, type in cv_link.items():
        class_paths = find_class_paths(type, onto_dict)
        class_paths = [os.path.split(cls)[-1] for cls in class_paths]
        for i in range(len(class_paths) - 1):
            if class_paths[i + 1] not in tree_dict:
                tree_dict[class_paths[i + 1]] = set()
            tree_dict[class_paths[i + 1]].add(class_paths[i])
    print('################## ontology tree ######################')
    print(tree_dict)
    root = os.path.split(THING_URI)[-1]
    queue = [set([root])]
    counter = 0
    while queue:
        layer_nums = len(queue)
        str = ''
        while layer_nums:
            node = queue.pop(0)
            str += '(' + ' '.join(node) + ')\t'
            for child in node:
                if child in tree_dict:
                    queue.append(tree_dict[child])
            layer_nums -= 1
        print(str)
        counter += 1
        if counter >= 10:
            break


##################################################################
######################### class complement #######################
##################################################################


def find_class_paths(type, onto_dict):
    subclass_dict = onto_dict['subclassof']
    # disjoint_dict = onto_dict['disjointwith']
    assert type in subclass_dict or type in subclass_dict.values()
    paths = [type]
    while (type in subclass_dict):
        paths.append(subclass_dict[type])
        type = subclass_dict[type]
    return paths


def calculate_freq(children, freq_dict):
    nums = 0
    ratio_dict = {}
    for c in children:
        nums += freq_dict[c]
    for c in children:
        ratio_dict[c] = freq_dict[c] / nums
    return ratio_dict


def calculate_entropy(children, freq_dict):
    if len(children) == 1:
        return 0
    else:
        # print(freq_dict)
        freq_dict = calculate_freq(children, freq_dict)
        freq_list = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)  # 根据频率排序，只取top 2计算熵
        # print('freq_list : ',freq_list)
        counter = 0
        entropy = 0
        # 将TOP2归一化后，然后计算熵
        nums = freq_list[0][1] + freq_list[1][1]
        top_1_prob = freq_list[0][1] * 1.0 / nums
        top_2_prob = freq_list[1][1] * 1.0 / nums
        top2_dict = {
            freq_list[0][0]: top_1_prob,
            freq_list[1][0]: top_2_prob
        }
        # print('top2_dict : ',top2_dict)
        for key, value in top2_dict.items():
            entropy += -1 * value * math.log(value, 2)
            counter += 1
            if counter >= 2:
                break
        return entropy


def get_ent_type(entity, cv_link):
    return cv_link[entity] if entity in cv_link else THING_URI


def find_class_by_relation(entity, cv_link, onto_dict,
                           h_rt, r_ht, t_hr,
                           entropy_threshold=0.7, loc='head'):
    relation = []
    if loc == 'head':
        if entity in h_rt:
            # entity作为头实体的relation
            relation = [r for r, t in h_rt[entity]]
        else:
            return {}
    elif loc == 'tail':
        if entity in t_hr:
            # entity作为尾实体的relation
            relation = [r for h, r in t_hr[entity]]
        else:
            return {}
    relation = list(set(relation))
    # 返回entity包含的三元组中另一个实体的Type
    other_ent_type = {}
    candidate_re = []
    if loc == 'head':
        candidate_re = [(r, t) for (r, t) in h_rt[entity]]
    elif loc == 'tail':
        candidate_re = [(r, h) for (h, r) in t_hr[entity]]
    for r, e in candidate_re:
        if r not in other_ent_type:
            other_ent_type[r] = set()
        if e in cv_link and cv_link[e] != THING_URI:
            t_type = cv_link[e]
            other_ent_type[r].add(t_type)
            # print(r,t,t_type,find_class_paths(t_type,onto_dict=onto_dict))
    # print(rel_t_type)
    # print('relation : ', relation)
    rel_type_prob = {}
    for rel in relation:
        rel_head_entities = set()
        # 找到每种关系它的尾实体类型，然后根据关系和尾实体类型推断出头实体类型
        candidate_ht = []
        if loc == 'head':
            candidate_ht = [(h, t) for (h, t) in r_ht[rel]]
        elif loc == 'tail':
            candidate_ht = [(t, h) for (h, t) in r_ht[rel]]
        # 将 h 定义为需要推断的class
        for h, t in candidate_ht:
            t_type = cv_link[t] if t in cv_link else THING_URI
            # 当前三元组的r和t均满足，则查看当前h是否是有效的class，也就是不是THING
            if t_type in other_ent_type[rel]:
                if get_ent_type(h, cv_link) != THING_URI:
                    rel_head_entities.add(h)
                    # print('yes : ',h,t,t_type,get_ent_type(h,cv_link))
        # rel_head_entities = set([h for (h,t) in r_ht[rel]])   # entity关系的头节点
        # print('rel head entities')
        # print(rel_head_entities)
        # 根据当前的r得到的头实体class推断出h可能的class类型
        rel_head_ontoligies = [cv_link[e] if e in cv_link else THING_URI for e in rel_head_entities]  # 头结点实体的本体
        class_freq = {}    # 记录每个type出现的次数
        class_tree_dict = {}  # 记录每个父节点和它的子节点
        for head_into in rel_head_ontoligies:
            class_paths = find_class_paths(head_into, onto_dict=onto_dict)
            # print(class_paths)
            for type in class_paths:
                class_freq[type] = class_freq.get(type, 0) + 1
            for i in range(len(class_paths) - 1):
                if class_paths[i + 1] not in class_tree_dict:
                    class_tree_dict[class_paths[i + 1]] = set()
                class_tree_dict[class_paths[i + 1]].add(class_paths[i])
        # print('class freq : ', class_freq)
        # print('class tree dict : ',class_tree_dict)
        # 根据频率或者熵筛选class，从根节点开始筛选
        flag = True
        node = THING_URI
        while flag:
            if node not in class_tree_dict:
                break
            children = class_tree_dict[node]
            # print('children : ', children)
            entropy = calculate_entropy(children, class_freq)
            # print('entropy : ',entropy)
            if entropy > entropy_threshold:  # 如果当前节点的熵小于阈值，则停止搜索
                flag = False
            else:    # 选择频率最高的children作为下一个搜索的节点
                children_freq = calculate_freq(children, class_freq)
                max_child = max(children_freq, key=class_freq.get)   # 获取最大值的child
                node = max_child
        # print('node : ', node)
        if node != THING_URI:
            if node not in rel_type_prob:
                rel_type_prob[node] = entropy
            rel_type_prob[node] = min(rel_type_prob[node], entropy)
    return rel_type_prob


def check_predict_type(result, data_info, index, supervised=False):
    new_result = []
    onto_dict = data_info["onto_dict"]
    ent_ills = data_info["ent_ills"]
    # total_nums = len(result)
    same_nums = 0
    contain_nums = 0
    conflict_nums = 0
    if index == 0:
        s2t_ills = dict(zip(ent_ills[0], ent_ills[1]))
        t_cv_link = data_info["lang2_cv_link"]
    else:
        s2t_ills = dict(zip(ent_ills[1], ent_ills[0]))
        t_cv_link = data_info["lang1_cv_link"]
    for ent, ent_type in result:
        t_ent = s2t_ills[ent]
        t_type = t_cv_link[t_ent]
        if type == t_type and type != THING_URI:
            # print('predict correct : ' ,type,t_type)
            same_nums += 1
            new_result.append((ent, ent_type))
        else:
            flag, ap, bp, _ = check_func(onto_dict, ent_type, t_type)
            if flag == 1:
                # print(type,t_type)
                contain_nums += 1
                new_result.append((ent, ent_type))
            else:
                conflict_nums += 1
                if supervised:
                    new_result.append((ent, ent_type))
                # print(ent,t_ent,type,t_type)
    return new_result


def calibrate_cv_link_by_seed(data_info, seed_ratio=0.3):
    ent_ills = data_info["ent_ills"]
    lang1_cv_link = data_info["lang1_cv_link"]
    lang2_cv_link = data_info["lang2_cv_link"]
    seed_nums = int(len(ent_ills[0]) * seed_ratio)
    # print('seed nums : ', seed_nums)
    change_nums = 0
    # print(ent_ills[0][:3],ent_ills[0][:3])
    for i in range(seed_nums):
        sent, tent = ent_ills[0][i],ent_ills[1][i]
        # print(sent,tent)
        if lang1_cv_link[sent] == THING_URI and lang2_cv_link[tent] != THING_URI:
            lang1_cv_link[sent] = lang2_cv_link[tent]
            change_nums += 1
        elif lang1_cv_link[sent] != THING_URI and lang2_cv_link[tent] == THING_URI:
            lang2_cv_link[tent] = lang1_cv_link[sent]
            change_nums += 1
    data_info["lang1_cv_link"] = lang1_cv_link
    data_info["lang2_cv_link"] = lang2_cv_link
    # print('change nums : {}'.format(change_nums))
    return data_info


def calibrate_cv_link_by_supervised_link(data_info, sup_links):
    # ent_ills = data_info["ent_ills"]
    lang1_cv_link = data_info["lang1_cv_link"]
    lang2_cv_link = data_info["lang2_cv_link"]
    change_nums = 0
    # print(sup_links[:3])
    for sent, tent in sup_links:
        if lang1_cv_link[sent] == THING_URI and lang2_cv_link[tent] != THING_URI:
            lang1_cv_link[sent] = lang2_cv_link[tent]
            change_nums += 1
        elif lang1_cv_link[sent] != THING_URI and lang2_cv_link[tent] == THING_URI:
            lang2_cv_link[tent] = lang1_cv_link[sent]
            change_nums += 1
    data_info["lang1_cv_link"] = lang1_cv_link
    data_info["lang2_cv_link"] = lang2_cv_link
    # print('change nums : {}'.format(change_nums))
    return data_info


def class_complement_by_relation(data_info, index, entropy, seed_ratio=1.0, supervised=True):
    # 利用监督种子补充缺失的class
    data_info = calibrate_cv_link_by_seed(data_info, seed_ratio)
    ############################
    kg = data_info["lang1_kg"] if index == 0 else data_info["lang2_kg"]
    cv_link = data_info["lang1_cv_link"] if index == 0 else data_info["lang2_cv_link"]
    onto_dict = data_info["onto_dict"]
    # ent_ills = data_info["ent_ills"]
    # miss class complement
    # counter = 0
    unknow_num = 0
    recall_num = 0
    # recall_correct_num = 0
    results = set()
    for ent, onto in cv_link.items():
        if onto == THING_URI:
            # print(ent,onto)
            unknow_num += 1
            # 根据ent分别作为头尾实体的class类型来决定ent的class
            # ent_as_head = kg['h_rt'][ent]
            # print(ent,onto)
            predict_as_head = find_class_by_relation(
                ent, cv_link, onto_dict, kg["h_rt"], kg["r_ht"], kg["t_hr"],
                entropy_threshold=entropy, loc='head')
            predict_as_tail = find_class_by_relation(
                ent, cv_link, onto_dict, kg["h_rt"], kg["r_ht"], kg["t_hr"],
                entropy_threshold=entropy, loc='tail')
            predict_result = dict(predict_as_head, **predict_as_tail)
            min_entropy = 1000
            predict_type = THING_URI
            for ent_type, entropy in predict_result.items():
                if min_entropy >= entropy:
                    min_entropy = entropy
                    predict_type = ent_type
            if min_entropy == 1000:
                continue
            else:
                recall_num += 1
                results.add((ent, predict_type))
            if index == 0:
                data_info["lang1_cv_link"][ent] = predict_type
            elif index == 1:
                data_info["lang2_cv_link"][ent] = predict_type
    # print('unknow nums : {} , recall num : {}'.format(unknow_num,recall_num))
    new_result = check_predict_type(results, data_info, index=index, supervised=supervised)
    # 根据填充结果修改class
    for ent, predict_type in new_result:
        if index == 0:
            data_info["lang1_cv_link"][ent] = predict_type
        elif index == 1:
            data_info["lang2_cv_link"][ent] = predict_type
    return data_info


def class_complement(data_info, index, entropy, sup_links=None, supervised=True):
    # 利用监督种子补充缺失的class
    if sup_links:
        data_info = calibrate_cv_link_by_supervised_link(data_info, sup_links)
    ############################
    kg = data_info["lang1_kg"] if index == 0 else data_info["lang2_kg"]
    cv_link = data_info["lang1_cv_link"] if index == 0 else data_info["lang2_cv_link"]
    onto_dict = data_info["onto_dict"]
    # ent_ills = data_info["ent_ills"]
    # miss class complement
    # counter = 0
    unknow_num = 0
    recall_num = 0
    # recall_correct_num = 0
    results = set()
    for ent, onto in cv_link.items():
        if onto == THING_URI:
            # print(ent,onto)
            unknow_num += 1
            # 根据ent分别作为头尾实体的class类型来决定ent的class
            # ent_as_head = kg['h_rt'][ent]
            # print(ent,onto)
            predict_as_head = find_class_by_relation(
                ent, cv_link, onto_dict, kg["h_rt"], kg["r_ht"], kg["t_hr"],
                entropy_threshold=entropy, loc='head')
            predict_as_tail = find_class_by_relation(
                ent, cv_link, onto_dict, kg["h_rt"], kg["r_ht"], kg["t_hr"],
                entropy_threshold=entropy, loc='tail')
            predict_result = dict(predict_as_head, **predict_as_tail)
            min_entropy = 1000
            predict_type = THING_URI
            for ent_type, entropy in predict_result.items():
                if min_entropy >= entropy:
                    min_entropy = entropy
                    predict_type = ent_type
            if min_entropy == 1000:
                continue
            else:
                recall_num += 1
                results.add((ent, predict_type))
            if index == 0:
                data_info["lang1_cv_link"][ent] = predict_type
            elif index == 1:
                data_info["lang2_cv_link"][ent] = predict_type
    # print('unknow nums : {} , recall num : {}'.format(unknow_num,recall_num))
    new_result = check_predict_type(results, data_info, index=index, supervised=supervised)
    # 根据填充结果修改class
    for ent, predict_type in new_result:
        if index == 0:
            data_info["lang1_cv_link"][ent] = predict_type
        elif index == 1:
            data_info["lang2_cv_link"][ent] = predict_type
    return data_info


def check_ills_consistent(onto_dict, ills):
    total_nums = len(ills)
    same_nums = 0
    contain_nums = 0
    conflict_nums = 0
    unsure_nums = 0
    for a, b in ills:
        flag, aParent, bParent, sign = check_func(onto_dict, a, b)
        if a == b and a != THING_URI:
            # print(aParent,bParent)
            same_nums += 1
        elif flag == 1:
            contain_nums += 1
        elif flag == -1:
            conflict_nums += 1
        elif a == THING_URI or b == THING_URI:
            unsure_nums += 1
    print('实体对总数:{}, class一致总数:{}, class属于包含关系总数:{}, class冲突总数:{}, class缺失总数:{}'.format(
        total_nums, same_nums, contain_nums, conflict_nums, unsure_nums))


##################################################################
#################### class path truncation #######################
##################################################################


def class_path_truncation(data_info, max_depth):
    onto_dict = data_info["onto_dict"]
    cv_link1 = data_info["lang1_cv_link"]
    cv_link2 = data_info["lang2_cv_link"]
    new_cv_link1, new_cv_link2 = {}, {}
    assert len(cv_link1) == len(cv_link2)
    for ent1, ent2 in zip(cv_link1.keys(), cv_link2.keys()):
        type1, type2 = cv_link1[ent1], cv_link2[ent2]
        flag, aParent, bParent, sign = check_func(onto_dict, type1, type2)
        if len(aParent) > max_depth:
            data_info["lang1_cv_link"][ent1] = aParent[(len(aParent)-max_depth)]
        if len(bParent) > max_depth:
            data_info["lang2_cv_link"][ent2] = bParent[(len(bParent)-max_depth)]
        new_cv_link1[ent1] = data_info["lang1_cv_link"][ent1]
        new_cv_link2[ent2] = data_info["lang2_cv_link"][ent2]
    # return {
    #     "ent_ills":data_info["ent_ills"],
    #     "lang1_cv_link":copy.deepcopy(data_info["lang1_cv_link"]),
    #     "lang2_cv_link":copy.deepcopy(data_info["lang2_cv_link"]),
    #     "lang1_kg":data_info["lang1_kg"],
    #     "lang2_kg":data_info["lang2_kg"],
    #     "onto_dict":data_info["onto_dict"]
    # }
    return new_cv_link1, new_cv_link2


def class_preprocess(folder, lang, supervised_links=None,
                     complement=True, trunction=True,
                     entropy_threshold=0.4,
                     epoch=5, supervised_filter=False,
                     seed_ratio=0.3, print_info=True, unify=1):
    data_info = load_datasets_info(folder, lang, unify)
    if not supervised_links:
        list1, list2 = data_info["ent_ills"]
        assert len(list1) == len(list2)
        sup_nums = int(len(list1) * seed_ratio)
        supervised_links = list(zip(list1, list2))[:sup_nums]
    # print("len sup links : {}".format(len(supervised_links)))
    if print_info:
        print('#################################################################')
        print('####################### class complement ########################')
        print('#################################################################')
        print('{}: 缺失class推断前:'.format(lang))
        type_ills = list(zip(data_info["lang1_cv_link"].values(), data_info["lang2_cv_link"].values()))
        check_ills_consistent(data_info["onto_dict"], type_ills)

    if complement:
        while epoch:
            data_info = class_complement(data_info,
                                         index=0,
                                         entropy=entropy_threshold,
                                         sup_links=supervised_links,
                                         supervised=supervised_filter)
            data_info = class_complement(data_info,
                                         index=1,
                                         entropy=entropy_threshold,
                                         sup_links=supervised_links,
                                         supervised=supervised_filter)
            epoch -= 1

        if print_info:
            print('{}:缺失class推断后:'.format(lang))
            type_ills = list(zip(data_info["lang1_cv_link"].values(), data_info["lang2_cv_link"].values()))
            check_ills_consistent(data_info["onto_dict"], type_ills)

    if trunction > 0:
        if print_info:
            print('#################################################################')
            print('##################### class path truncation #####################')
            print('#################################################################')

        cv_link1, cv_link2 = class_path_truncation(data_info, max_depth=trunction)
        # assert data_info["lang1_cv_link"] == new_data_info["lang1_cv_link"]
        if print_info:
            type_ills = list(zip(cv_link1.values(), cv_link2.values()))
            print('允许最大深度:{}'.format(trunction), end='\t')
            check_ills_consistent(data_info["onto_dict"], type_ills)
    return data_info["lang1_cv_link"], data_info["lang2_cv_link"]
    # return cv_link1,cv_link2


if __name__ == '__main__':
    lang = "zh_en"
    DBP15K_RAW_PATH = ''
    data_info = load_datasets_info(DBP15K_RAW_PATH, lang, unify=1)

    print('#################################################################')
    print('####################### class complement ########################')
    print('#################################################################')
    epoch = 5
    print('{} : 缺失class推断前： '.format(lang))
    type_ills = list(zip(data_info["lang1_cv_link"].values(), data_info["lang2_cv_link"].values()))
    # print(len(type_ills),type_ills[:2])
    check_ills_consistent(data_info["onto_dict"], type_ills)
    while epoch:
        seed_ratio = 0.3
        data_info = class_complement_by_relation(data_info, index=0,
                                                 entropy=0.4,
                                                 seed_ratio=seed_ratio,
                                                 supervised=False)
        data_info = class_complement_by_relation(data_info, index=1,
                                                 entropy=0.4,
                                                 seed_ratio=seed_ratio,
                                                 supervised=False)
        epoch -= 1
    print('{}:缺失class推断后:'.format(lang))
    type_ills = list(zip(data_info["lang1_cv_link"].values(), data_info["lang2_cv_link"].values()))
    check_ills_consistent(data_info["onto_dict"], type_ills)

    print('#################################################################')
    print('##################### class path truncation #####################')
    print('#################################################################')

    max_depths = [6, 5, 4, 3]
    for md in max_depths:
        cv_link1, cv_link2 = class_path_truncation(data_info, max_depth=md)
        # assert data_info["lang1_cv_link"] == new_data_info["lang1_cv_link"]
        type_ills = list(zip(cv_link1.values(), cv_link2.values()))
        print('允许最大深度：{}'.format(md), end='\t')
        check_ills_consistent(data_info["onto_dict"], type_ills)

    ################  test ####################
    class_preprocess(DBP15K_RAW_PATH, lang, complement=False, trunction=4)
