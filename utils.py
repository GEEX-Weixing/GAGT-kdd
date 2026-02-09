import torch
import torch.nn.modules.loss
import torch.nn.functional as F
import scipy.sparse as sp
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import random
from torch_geometric.data import Data

def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def gamma_re_loss(x_orig, x_recon, gamma):
    """
    实现论文公式 (3): L_rec = mean( max( ||x - x_recon||_2 - gamma, 0 ) )
    """
    # 计算每个样本的 L2 范数（对非 batch 维度求和）
    l2_norm = torch.norm(x_orig - x_recon, p=2, dim=tuple(range(1, x_orig.dim())))

    loss = torch.clamp(l2_norm - gamma, min=0.0)

    # 返回平均损失
    return loss.mean()

def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_graph(adj, device):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized).to(device)

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])
    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def for_gae(features, adj, device):
    n_nodes, feat_dim = features.shape
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj, device)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    return adj_label.to(device), torch.tensor(norm).to(device), torch.tensor(pos_weight).to(device)

def compute_accuracy_teacher_mask(prediction, label, mask):
    correct = 0
    indices = torch.nonzero(mask)
    for i in indices:
        if prediction[i] == label[i]:
            correct += 1
    accuracy = correct / len(prediction) * 100
    return accuracy

def compute_accuracy_teacher(prediction, label):
    correct = 0
    # label = torch.argmax(label, dim=1)
    for i in range(len(label)):
        if prediction[i] == label[i]:
            correct += 1
    accuracy = correct / len(prediction) * 100
    return accuracy

import torch.nn as nn
from torch import Tensor


class SemanticConsistency(nn.Module):
    def __init__(self, ignore_index=(), reduction='mean'):
        super(SemanticConsistency, self).__init__()
        self.ignore_index = ignore_index
        self.loss = nn.CrossEntropyLoss(ignore_index=-1, reduction=reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        for class_idx in self.ignore_index:
            target[target == class_idx] = -1
        return self.loss(input, target)


def entropy_loss_f(logit):
    probs_t = F.softmax(logit, dim=-1)
    probs_t = torch.clamp(probs_t, min=1e-9, max=1.0)
    entropy_loss = torch.mean(torch.sum(-probs_t * torch.log(probs_t), dim=-1))
    return entropy_loss

def intra_domain_contrastive_loss(z, z_1, z_2, decoder, num_shuffles=60, shuffle_batch_size=20,
                                    temperature=0.5, generator=None, domain_predictor=None,
                                    domain_labels=None, weight_min=0.15):
    N, D_z = z.shape
    device = z.device

    # ==================== 自适应权重计算(核心新增) ==========
    if domain_predictor is not None and domain_labels is not None:
        with torch.no_grad():
            # 预测域概率分布
            domain_probs = F.softmax(domain_predictor(z.detach()), dim=-1)  # [N, num_domains]

            # 提取每个样本对应其真实域的置信度
            domain_labels = domain_labels.to(device)
            gt_weights = domain_probs.gather(1, domain_labels.unsqueeze(1)).squeeze(1)  # [N]

            # 应用下限约束: max(weight_min, gt_weight)
            dynamic_weights = torch.clamp(gt_weights, min=weight_min)

            # 转换为1/weight形式（类比原始MAE的1/gt_weight）
            adaptive_weights = 1.0 / dynamic_weights  # [N]
    else:
        # 不使用自适应权重时，所有权重为1.0
        adaptive_weights = torch.ones(N, device=device)

    # ========== 原始逻辑不变 ==========
    z_norm = F.normalize(z, dim=1)

    with torch.no_grad() if shuffle_batch_size > 0 else torch.enable_grad():
        _, pos_recon = decoder(torch.cat((z_1, z_2), 1))
    pos_recon_norm = F.normalize(pos_recon.detach(), dim=1)
    pos_sim = torch.sum(z_norm * pos_recon_norm, dim=1) / temperature

    total_loss = 0.0
    num_batches = (num_shuffles + shuffle_batch_size - 1) // shuffle_batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * shuffle_batch_size
        end = min(start + shuffle_batch_size, num_shuffles)
        current_batch_size = end - start

        # 生成负样本
        neg_sims_list = []
        for _ in range(current_batch_size):
            perm = torch.randperm(N, device=device, generator=generator)
            z2_shuffled = z_2[perm]
            _, neg_recon = decoder(torch.cat((z_1, z2_shuffled), 1))
            neg_recon_norm = F.normalize(neg_recon, dim=1)
            neg_sim = torch.sum(z_norm * neg_recon_norm, dim=1) / temperature
            neg_sims_list.append(neg_sim)

        neg_sims = torch.stack(neg_sims_list, dim=1)
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1)
        labels = torch.zeros(N, dtype=torch.long, device=device)

        # ========== 关键修改：加权损失 ====================
        # 计算每个样本的损失（不使用reduction='mean'）
        loss_per_sample = F.cross_entropy(logits, labels, reduction='none')  # [N]
        # 应用自适应权重
        weighted_loss = (loss_per_sample * adaptive_weights).mean()
        total_loss += weighted_loss

    avg_loss = total_loss / num_batches
    return avg_loss

