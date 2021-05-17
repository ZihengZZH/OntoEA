import numpy as np
from openea.modules.finding.alignment import greedy_alignment


def valid(embeds1, embeds2, mapping, top_k, threads_num, metric='inner',
          normalize=False, csls_k=0, accurate=False,
          type_mat=None, type_weight=0.2, enum_weight=False):
    if mapping is None:
        _, hits1_12, mr_12, mrr_12 = greedy_alignment(
            embeds1, embeds2, top_k, threads_num,
            metric, normalize, csls_k, accurate, type_mat=type_mat, type_weight=type_weight,
            enum_weight=enum_weight)
    else:
        test_embeds1_mapped = np.matmul(embeds1, mapping)
        _, hits1_12, mr_12, mrr_12 = greedy_alignment(
            test_embeds1_mapped, embeds2, top_k, threads_num,
            metric, normalize, csls_k, accurate, type_mat=type_mat, type_weight=type_weight,
            enum_weight=enum_weight)
    return hits1_12, mrr_12


def test(embeds1, embeds2, mapping, top_k, threads_num, metric='inner',
         normalize=False, csls_k=0, accurate=True, onto_eval=False,
         type_mat=None, type_weight=0.2, enum_weight=False):
    if mapping is None:
        alignment_rest_12, hits1_12, mr_12, mrr_12 = greedy_alignment(
            embeds1, embeds2, top_k, threads_num,
            metric, normalize, csls_k, accurate, onto_eval, type_mat, type_weight,
            enum_weight=enum_weight)
    else:
        test_embeds1_mapped = np.matmul(embeds1, mapping)
        alignment_rest_12, hits1_12, mr_12, mrr_12 = greedy_alignment(
            test_embeds1_mapped, embeds2, top_k, threads_num,
            metric, normalize, csls_k, accurate, onto_eval, type_mat, type_weight,
            enum_weight=enum_weight)
    return alignment_rest_12, hits1_12, mrr_12


def early_stop(flag1, flag2, flag):
    if flag <= flag2 <= flag1:
        print("\n == should early stop == \n")
        return flag2, flag, True
    else:
        return flag2, flag, False
