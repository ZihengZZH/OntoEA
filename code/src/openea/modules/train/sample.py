
import random
import numpy as np
import time


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)


def prob_pick(lists, probs):
    x = random.uniform(0, 1)
    cumu_prob = 0.0
    result = random.sample(lists, 1)[0]
    for v, p in zip(lists, probs):
        cumu_prob += p
        if x < cumu_prob:
            result = v
            break
    return result


def prob_sample(lists, probs, nums, max_reply=10):
    results = set()
    while nums:
        counter = 0
        while counter < max_reply:
            item = prob_pick(lists, probs)
            if item not in results:
                results.add(item)
                break
            else:
                counter += 1
        nums -= 1
    counter = 0
    while len(results) < nums or counter > max_reply:
        results |= set(random.sample(lists, nums-len(results)))
        counter += 1
    # print("sample result : ",results)
    return list(results)


def generate_ent_prob_dict(ent2onto_dict, entities_lists, onto_mat):
    t1 = time.time()
    ontologies_lists = []
    dic = {}
    for ent in entities_lists:
        print(ent)
        ontologies_lists.append(ent2onto_dict[ent])
    onto_dic = {}
    for onto in onto_mat.keys():
        onto_dic[onto] = [onto_mat[onto][i] for i in ontologies_lists]
    for ent in entities_lists:
        onto = ent2onto_dict[ent]
        dic[ent] = onto_dic[onto]
    del onto_dic
    print("\ngenerating {} entity prob dict for neg sample costs {:.3f} s.".format(len(entities_lists), time.time() - t1))
    return dic
