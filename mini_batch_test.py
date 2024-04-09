import numpy as np
import math
from metrics import IDCG,GINI
import faiss
import numba as nb

@nb.njit(nopython=True)
def compute_ranking_metrics(testusers, testdata, traindata, user_rank_pred_items,topk=20):
    all_metrics = []
    for i in range(len(testusers)):
        u = testusers[i]
        one_metrics = []
        test_items = testdata[i]
        pos_length = len(test_items)
        # pred_items找出ranking结果中排除训练样本的前topk个items
        mask_items = traindata[i]
        pred_items_all = user_rank_pred_items[u]
        max_length_candicate = len(mask_items) + topk
        pred_items = [item for item in pred_items_all[:max_length_candicate]
                      if item not in mask_items][:topk]
        hit_value = 0
        dcg_value = 0
        for idx in range(topk):
            if pred_items[idx] in test_items:
                hit_value += 1
                dcg_value += math.log(2) / math.log(idx + 2)
        target_length = min(topk, pos_length)
        idcg = 0.0
        for i in range(target_length):
            idcg = idcg + math.log(2) / math.log(i + 2)
        hr_cur = hit_value / target_length
        recall_cur = hit_value / pos_length
        ndcg_cur = dcg_value / idcg
        one_metrics.append([hr_cur, recall_cur, ndcg_cur])
        all_metrics.append(one_metrics)
    return all_metrics


def num_faiss_evaluate(_test_ratings, _train_ratings,  _user_matrix, _item_matrix,Topk=20):
    '''
    Evaluation for ranking results
    Topk-largest based on faiss search
    Speeding computation based on numba
    '''

    ###  faiss search  ###
    
    query_vectors = _user_matrix
    test_users = list(_test_ratings.keys())
    dim = _user_matrix.shape[-1]
    index = faiss.IndexFlatIP(dim)
    index.add(_item_matrix)
    max_mask_items_length = max(
        len(_train_ratings[user]) for user in _train_ratings.keys())
    sim, _user_rank_pred_items = index.search(
        query_vectors, Topk+max_mask_items_length)

    testdata = [_test_ratings[user] for user in _test_ratings.keys()]
    traindata = [_train_ratings[user] for user in _test_ratings.keys()]
    all_metrics = compute_ranking_metrics(nb.typed.List(test_users), nb.typed.List(testdata),
                                          nb.typed.List(traindata), nb.typed.List(_user_rank_pred_items),topk=Topk)

    all_metrics=np.array(all_metrics).T
    hr_out=np.sum(all_metrics[0])
    recall_out=np.sum(all_metrics[1])
    ndcg_out=np.sum(all_metrics[2])


    return hr_out,recall_out,ndcg_out


def test_acc_batch(_test_U2I, _train_U2I,  _user_matrix, _item_matrix,topk=20):
    test_users = list(_test_U2I.keys())
    batch_id = 0
    batch_size=35000
    data_size = len(test_users)
    hr_all,recall_all,ndcg_all=0.0,0.0,0.0
    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            batch_users=[test_users[idx] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            batch_users=[test_users[idx] for idx in range(batch_id, data_size)]
            batch_id = data_size

        hr_out,recall_out,ndcg_out=num_faiss_evaluate({key: _test_U2I[key] for key in batch_users},{key: _train_U2I[key] for key in batch_users},_user_matrix,_item_matrix,Topk=topk)
        hr_all=hr_all+hr_out
        recall_all=recall_all+recall_out
        ndcg_all=ndcg_all+ndcg_out
    hr_all=hr_all/data_size
    recall_all=recall_all/data_size
    ndcg_all=ndcg_all/data_size
    return hr_all,recall_all,ndcg_all
