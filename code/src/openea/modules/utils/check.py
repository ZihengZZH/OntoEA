import warnings
import sys
import os
from tqdm import tqdm
import numpy as np
import h5py


warnings.filterwarnings('ignore')
sys.path.append("../")
THING_URI = 'http://www.w3.org/2002/07/owl#Thing'


'''
构建冲突矩阵
'''


def is_thing(line):
    if THING_URI in line:
        return True
    return False


def load_type_relation(dataset_division, bracket=True):
    subclass_path = os.path.join(dataset_division, 'onto_subClassOf_triples')
    disjoint_path = os.path.join(dataset_division, 'onto_disjointWith_triples')
    subclass_dict = {}
    disjoint_dict = {}
    assert os.path.isfile(subclass_path)
    assert os.path.isfile(disjoint_path)
    # read subClassOf
    with open(subclass_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            item = line.strip().split()
            assert len(item) >= 3
            if bracket:
                item = [i[1:-1] for i in item]
            subclass_dict[item[0]] = item[2]
    # read disjointWith
    with open(disjoint_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            item = line.strip().split()
            assert len(item) >= 3
            if bracket:
                item = [i[1:-1] for i in item]
            disjoint_dict[item[0]] = item[2]
    return {
        'subclassof': subclass_dict,
        'disjointwith': disjoint_dict
    }


def check_type_coherence(onto_dict, dataset_division,
                         unsure_w=0, version=0,
                         save_name='onto_check_mat.h5', k=0):
    ontologies = list(set(onto_dict['subclassof'].keys()) | set(onto_dict['subclassof'].values()))
    onto_id = {}
    onto_num = len(ontologies)
    for i in range(onto_num):
        onto_id[ontologies[i]] = i
    onto_isA_mat = np.zeros((onto_num, onto_num))
    print('onto isA mat shape : ', onto_isA_mat.shape)
    try:
        with tqdm(range(onto_num), ncols=150) as t:
            for i in t:
                for j in range(onto_num):
                    # print(ontologies[i],ontologies[j])
                    if version == 0:
                        flag, _, _ = check_type_v1(onto_dict, ontologies[i], ontologies[j])
                    elif version == 1:
                        flag, _, _, _ = check_type_v2(onto_dict, ontologies[i], ontologies[j], unsure_w=unsure_w)
                    elif version == 2:
                        # 相比v1，考虑了公共父节点的情况，例如（音乐家，贵族）
                        flag, _, _ = check_type_v3(onto_dict, ontologies[i], ontologies[j])
                    elif version == 3:
                        # 只有完全相等才为1，也就是矩阵只有对角线为1
                        flag, _, _ = check_type_v4(onto_dict, ontologies[i], ontologies[j])
                    elif version == 4:
                        # 利用路径集合的交并集比例作为权重
                        flag, _, _ = check_type_v5(onto_dict, ontologies[i], ontologies[j], k)
                    elif version == 5:
                        # 利用路径的交并集比例作为权重
                        flag, _, _ = check_type_v6(onto_dict, ontologies[i], ontologies[j], k)
                    onto_isA_mat[i][j] = flag
    except KeyboardInterrupt:
        t.close()
        raise
    t.close()
    print(onto_isA_mat)
    # save2file
    file = h5py.File(os.path.join(dataset_division, save_name), 'w')
    file.create_dataset('onto_mat', data=onto_isA_mat)
    onto_name = [n.encode('utf-8') for n in list(onto_id.keys())]
    file.create_dataset('onto_name', data=onto_name)
    file.create_dataset('onto_id', data=list(onto_id.values()))
    file.close()
    return onto_isA_mat


def check_type_v2(onto_dict, a, b, unsure_w=0.5):
    '''
    return 1 : 两个实体的type是一致的
    return -1 : type不一致
    return 0 : 不确定，例如两个type都是thing
    '''
    subclass_dict = onto_dict['subclassof']
    disjoint_dict = onto_dict['disjointwith']
    assert a in subclass_dict or a in subclass_dict.values()
    assert b in subclass_dict or b in subclass_dict.values()
    aParent, bParent = [a], [b]
    while (a in subclass_dict):
        aParent.append(subclass_dict[a])
        a = subclass_dict[a]
    while (b in subclass_dict):
        bParent.append(subclass_dict[b])
        b = subclass_dict[b]

    aLen, bLen = len(aParent), len(bParent)
    # print('a parent : ',aParent)
    # print('b parent : ',bParent)

    if check_disjoint(disjoint_dict, aParent, bParent):     # 检查两个父类列表包含冲突本体
        # print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # print('a parent : ',aParent)
        # print('b parent : ',bParent)
        return -1, aParent, bParent, 'disjoint'

    if aLen == bLen == 1 and aParent[0] == THING_URI:   # 如果两个本体都是Thing
        return unsure_w, aParent, bParent, 'allthing'
    # 从后往前找公共父类链表
    start1, start2 = aLen - 1, bLen - 1
    while start1 >= 0 and start2 >= 0:
        if aParent[start1] != bParent[start2]:
            break
        start1 -= 1
        start2 -= 1
    # print(start1,start2)
    if start1 < 0 or start2 < 0:  # 包含关系
        if start1 < 0 and aParent[start1 + 1] == THING_URI:
            flag = unsure_w
        elif start2 < 0 and bParent[start2 + 1] == THING_URI:
            flag = unsure_w
        else:
            flag = 1
    else:    # 可能是兄弟结点
        if aParent[start1+1] == THING_URI:   # 如果公共结点是-1
            flag = -1
        else:
            flag = 1

    # if aLen >= bLen:    # 从后往前找
    #     flag = 1 if (aParent[0] == bParent[0]) and not is_thing(aParent[0]) else -1
    # elif aLen > bLen:
    #     flag = 1 if (aParent[aLen-bLen] == bParent[0]) and not is_thing(bParent[0]) else -1
    # else:
    #     flag = 1 if (bParent[bLen - aLen] == aParent[0]) and not is_thing(aParent[0]) else -1
    if flag == -1:
        # print(abs(aLen-bLen))
        # if abs(aLen-bLen) == 4:
        #     print('a parent : ',aParent)
        #     print('b parent : ',bParent)
        # print('a parent : ', aParent)
        # print('b parent : ', bParent)
        pass
    return flag, aParent, bParent, 'normal'


def check_disjoint(onto_dict, aList, bList):
    flag = 0
    for a in aList:
        if a in onto_dict:
            if onto_dict[a] in bList:
                flag = 1
    for b in bList:
        if b in onto_dict:
            if onto_dict[b] in aList:
                flag = 1
    return flag


# 第一个版本
def check_type_v1(onto_dict, a, b):
    '''
    return 1 : 两个实体的type是一致的
    return -1 : type不一致
    return 0 : 不确定，例如两个type都是thing
    '''
    subclass_dict = onto_dict['subclassof']
    # disjoint_dict = onto_dict['disjointwith']
    assert a in subclass_dict or a in subclass_dict.values()
    assert b in subclass_dict or b in subclass_dict.values()
    aParent, bParent = [a], [b]
    while (a in subclass_dict):
        aParent.append(subclass_dict[a])
        a = subclass_dict[a]
    while (b in subclass_dict):
        bParent.append(subclass_dict[b])
        b = subclass_dict[b]
    aLen, bLen = len(aParent), len(bParent)
    if aLen == bLen == 1 and aParent[0] == THING_URI:
        return 0.5, aParent, bParent
    if aLen == bLen:
        flag = 1 if (aParent[0] == bParent[0]) and not is_thing(aParent[0]) else 0
    elif aLen > bLen:
        flag = 1 if (aParent[aLen-bLen] == bParent[0]) and not is_thing(bParent[0]) else 0
    else:
        flag = 1 if (bParent[bLen - aLen] == aParent[0]) and not is_thing(aParent[0]) else 0
    return flag, aParent, bParent


def check_type_v3(onto_dict, a, b):
    '''
    相比v2，考虑了公共父类的情况
    return 1 : 两个实体的type是一致的
    return -1 : type不一致
    return 0 : 不确定，例如两个type都是thing
    '''
    subclass_dict = onto_dict['subclassof']
    # disjoint_dict = onto_dict['disjointwith']
    assert a in subclass_dict or a in subclass_dict.values()
    assert b in subclass_dict or b in subclass_dict.values()
    aParent, bParent = [a], [b]
    while (a in subclass_dict):
        aParent.append(subclass_dict[a])
        a = subclass_dict[a]
    while (b in subclass_dict):
        bParent.append(subclass_dict[b])
        b = subclass_dict[b]
    aLen, bLen = len(aParent), len(bParent)
    # print('a parent : ',aParent)
    # print('b parent : ',bParent)
    # if aLen == bLen == 1 and aParent[0] == THING_URI:
    #     return 1,aParent,bParent
    if aLen == bLen:
        if (aParent[0] == bParent[0]):   # 相等
            flag = 1
        elif (aParent[1] == bParent[1]) and not is_thing(aParent[1]):   # 具有公共父节点
            flag = 1
        else:
            flag = 0
    elif aLen > bLen:
        flag = 1 if (aParent[aLen-bLen] == bParent[0]) and not is_thing(bParent[0]) else 0
    else:
        flag = 1 if (bParent[bLen - aLen] == aParent[0]) and not is_thing(aParent[0]) else 0
    # if flag == -1:
    #     print('a parent : ',aParent)
    #     print('b parent : ',bParent)
    return flag, aParent, bParent


def check_type_v4(onto_dict, a, b):
    '''
    只有相等才为1
    return 1 : 两个实体的type是一致的
    return -1 : type不一致
    return 0 : 不确定，例如两个type都是thing
    '''
    subclass_dict = onto_dict['subclassof']
    # disjoint_dict = onto_dict['disjointwith']
    assert a in subclass_dict or a in subclass_dict.values()
    assert b in subclass_dict or b in subclass_dict.values()
    aParent, bParent = [a], [b]
    while (a in subclass_dict):
        aParent.append(subclass_dict[a])
        a = subclass_dict[a]
    while (b in subclass_dict):
        bParent.append(subclass_dict[b])
        b = subclass_dict[b]
    aLen, bLen = len(aParent), len(bParent)
    if aLen == bLen and aParent[0] == bParent[0]:
        flag = 1
    else:
        flag = 0
    return flag, aParent, bParent


def check_type_v5(onto_dict, a, b, k=0):
    '''
    基于集合的交集和并集计算权重
    return 1 : 两个实体的type是一致的
    return -1 : type不一致
    return 0 : 不确定，例如两个type都是thing
    '''
    subclass_dict = onto_dict['subclassof']
    disjoint_dict = onto_dict['disjointwith']
    assert a in subclass_dict or a in subclass_dict.values()
    assert b in subclass_dict or b in subclass_dict.values()
    aParent, bParent = [a], [b]
    while (a in subclass_dict):
        aParent.append(subclass_dict[a])
        a = subclass_dict[a]
    while (b in subclass_dict):
        bParent.append(subclass_dict[b])
        b = subclass_dict[b]

    if check_disjoint(disjoint_dict, aParent, bParent):     # 检查两个父类列表包含冲突本体
        weight = 0
    else:
        aLen, bLen = len(aParent), len(bParent)
        if aLen == bLen and aParent[0] == bParent[0]:   # 相等
            weight = 1
        else:
            aset, bset = set(aParent), set(bParent)
            inter_num = len(aset & bset)
            union_num = len(aset)+len(bset)
            weight = (inter_num*2+k)/(union_num+k)
    return weight, aParent, bParent


def check_type_v6(onto_dict, a, b, k=0):
    '''
    基于路径的交集和并集计算权重
    return 1 : 两个实体的type是一致的
    return -1 : type不一致
    return 0 : 不确定，例如两个type都是thing
    '''
    subclass_dict = onto_dict['subclassof']
    disjoint_dict = onto_dict['disjointwith']
    assert a in subclass_dict or a in subclass_dict.values()
    assert b in subclass_dict or b in subclass_dict.values()
    aParent, bParent = [a], [b]
    while (a in subclass_dict):
        aParent.append(subclass_dict[a])
        a = subclass_dict[a]
    while (b in subclass_dict):
        bParent.append(subclass_dict[b])
        b = subclass_dict[b]

    if check_disjoint(disjoint_dict, aParent, bParent):     # 检查两个父类列表包含冲突本体
        weight = 0
    else:
        aLen, bLen = len(aParent), len(bParent)
        if aLen == bLen and aParent[0] == bParent[0]:   # 相等
            weight = 1
        else:
            aset, bset = set(aParent), set(bParent)
            inter_num = len(aset & bset)
            union_num = max(len(aset), len(bset))
            weight = (inter_num+k)/(union_num+k)
    return weight, aParent, bParent
