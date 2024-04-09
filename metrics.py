import torch
import pandas as pd
import numpy as np
import math
from scipy.stats import spearmanr
import torch.nn.functional as F
import faiss

def GINI(list, num):
    se = pd.Series(list)
    value = se.value_counts(normalize=True, ascending=True).values.tolist()
    diff = num - len(value)
    sorted_x = np.sum(np.cumsum(np.array([0] * diff + value)))
    gini = (num + 1 - 2 * sorted_x) / num
    return gini


def IDCG(n):
    idcg = 0
    for i in range(n):
        idcg += 1 / math.log(i + 2, 2)
    return idcg


def PRU(pop_list, rank_list):
    if sum(np.array(pop_list) == pop_list[0]) == len(pop_list):
        pop_list[-1] = pop_list[-1] + 1e-5
    if sum(np.array(rank_list) == rank_list[0]) == len(rank_list):
        rank_list[-1] = rank_list[-1] + 1e-5
    stats, p = spearmanr(pop_list, rank_list)
    pru = (-1) * stats
    return pru


def InfoNCE(view1, view2, temperature):
    view1, view2 = torch.nn.functional.normalize(
        view1, dim=1), torch.nn.functional.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / ttl_score)
    return torch.mean(cl_loss)

def InfoNCE_i(view1, view2, view3,temperature,gama):
    view1, view2,view3 = torch.nn.functional.normalize(
        view1, dim=1), torch.nn.functional.normalize(view2, dim=1), torch.nn.functional.normalize(view3, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score_1 = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score_1 = torch.exp(ttl_score_1 / temperature).sum(dim=1)
    ttl_score_2 = torch.matmul(view1, view3.transpose(0, 1))
    ttl_score_2 = torch.exp(ttl_score_2 / temperature).sum(dim=1)

    cl_loss = -torch.log(pos_score / (gama*ttl_score_2+ttl_score_1+pos_score))
    return torch.mean(cl_loss)

     


# def super_InfoNCE(view1,view2,view3,temperature):
#     view1, view2,view3 = torch.nn.functional.normalize(view1, dim=1), torch.nn.functional.normalize(view2, dim=1), torch.nn.functional.normalize(view3, dim=1)
#     pos_score = (view1 * view2).sum(dim=-1)
#     pos_score = torch.exp(pos_score / temperature)
#     ttl_score = torch.matmul(view1, view3.transpose(0, 1))
#     ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
#     cl_loss = -torch.log(pos_score / ttl_score)
#     return torch.sum(cl_loss)

def super_InfoNCE(view1_1,view2_1,view3_1,temperature):
    view1, view2,view3 = torch.nn.functional.normalize(view1_1, dim=1), torch.nn.functional.normalize(view2_1, dim=1), torch.nn.functional.normalize(view3_1, dim=1)
    # Compute similarity between anchor and positives
    pos_sim = torch.matmul(view1, view2.transpose(0, 1))
    pos_sim=torch.exp(pos_sim / temperature).sum(1)
    # Compute similarity between anchor and negatives
    neg_sim = torch.matmul(view1, view3.transpose(0, 1))
    neg_sim=torch.exp(neg_sim / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_sim / (pos_sim+neg_sim))
    return torch.sum(cl_loss)

def euclidean_dist(x, y):
        """
        Args:
          x: pytorch Variable, with shape [m, d]
          y: pytorch Variable, with shape [n, d]
        Returns:
          dist: pytorch Variable, with shape [m, n]
        """
 
        m, n = x.size(0), y.size(0)
        # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        # yy会在最后进行转置的操作
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT 
        dist.addmm_(1, -2, x, y.t())
        # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)+1e-8) / _range


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """计算Gram核矩阵
    source: sample_size_1 * feature_size 的数据
    target: sample_size_2 * feature_size 的数据
    kernel_mul: 这个概念不太清楚，感觉也是为了计算每个核的bandwith
    kernel_num: 表示的是多核的数量
    fix_sigma: 表示是否使用固定的标准差
        return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的
                        矩阵，表达形式:
                        [   K_ss K_st
                            K_ts K_tt ]
    """
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0) # 合并在一起

    total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2) # 计算高斯核中的|x-y|

    # 计算多核中每个核的bandwidth
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    # 高斯核的公式，exp(-|x-y|/bandwith)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val) # 将多个核合并在一起

def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n = int(source.size()[0])
    m = int(target.size()[0])

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:n, :n] 
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]

    XX = torch.div(XX, n * n).sum(dim=1).view(1,-1)  # K_ss矩阵，Source<->Source
    XY = torch.div(XY, -n * m).sum(dim=1).view(1,-1) # K_st矩阵，Source<->Target

    YX = torch.div(YX, -m * n).sum(dim=1).view(1,-1) # K_ts矩阵,Target<->Source
    YY = torch.div(YY, m * m).sum(dim=1).view(1,-1)  # K_tt矩阵,Target<->Target
    	
    loss = (XX + XY).sum() + (YX + YY).sum()
    return loss

def uniformity(x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()   


#user侧的指标
def DAC_user(degree_list,acc_list):
     stats,_= spearmanr(degree_list, acc_list) 
     return (-1)*stats

def overall_uniform(embedding,t=2):
        x = F.normalize(torch.tensor(embedding), dim=-1)
        return torch.pdist(torch.tensor(embedding), p=2).pow(2).mul(-t).exp().mean().log()

def overall_uniform_weight(index_list, embedding):
    """ Args:
    index_list (torch.LongTensor): user/item ids of positive interactions, shape: [|R|, ]
    embedding (torch.nn.Embedding): user/item embeddings, shape: [|U|, dim] or [|I|, dim]
    """ 
    values, _= torch.sort(index_list)
    count_series = pd.value_counts(values.tolist(), sort=False)
    count = torch.from_numpy(count_series.values).unsqueeze(0)

    weight_matrix = torch.mm(count.transpose(-1, 0), count)
    weight = torch.triu(weight_matrix, 1).view(-1)[
        torch.nonzero(torch.triu(weight_matrix, 1).view(-1)).view(-1)].to(embedding.device)
    total_freq = (len(index_list) * len(index_list) - weight_matrix.trace()) / 2
    return torch.pdist(embedding[count_series.index], p=2).pow(2).mul(-2).exp().mul(weight).sum().div(total_freq).log().item()

def run_kmeans(x, num_cluster,device):
    """Run K-means algorithm to get k clusters of the input tensor x"""
    kmeans = faiss.Kmeans(d=64, k=num_cluster, gpu=device)
    kmeans.train(x)
    cluster_cents = kmeans.centroids
    _, I = kmeans.index.search(x, 1)
    return cluster_cents, I
