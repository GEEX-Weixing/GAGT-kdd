import numpy as np
import scipy.sparse as sp
import scipy.io as sio
from scipy.sparse import lil_matrix
import scipy
import torch
from torch_geometric.data import Data
from scipy.sparse import csc_matrix

def ComputePPMI(A):
    '''compute PPMI, given aggregated K-step transition probality matrix as input'''
    np.fill_diagonal(A, 0)
    A = MyScaleSimMat(A)
    (p, q) = np.shape(A)
    col = np.sum(A, axis=0)
    col[col == 0] = 1
    PPMI = np.log((float(p) * A) / col[None, :])
    IdxNan = np.isnan(PPMI)
    PPMI[IdxNan] = 0
    PPMI[PPMI < 0] = 0
    return PPMI

def MyScaleSimMat(W):
    '''L1 row norm of a matrix'''
    rowsum = np.array(np.sum(W, axis=1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    W = r_mat_inv.dot(W)
    return W

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def AggTranProbMat(G, step):
    '''aggregated K-step transition probality'''
    G = MyScaleSimMat(G)
    G = csc_matrix.toarray(G)
    A_k = G
    A = G
    for k in np.arange(2, step + 1):
        A_k = np.matmul(A_k, G)
        A = A + A_k / k
    return A

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    mx = mx.todense()
    # rowsum = np.array(mx.sum(1),dtype=np.float32)
    # r_inv = np.power(rowsum, -1).flatten()
    # r_inv[np.isinf(r_inv)] = 0.
    # r_mat_inv = sp.diags(r_inv)
    # mx = r_mat_inv.dot(mx)
    mx = np.array(mx)
    # row_sums = mx.sum(axis=1, keepdims=True)  # [N, 1]
    # row_sums_safe = np.where(row_sums == 0, 1, row_sums)
    # normalized_mx = mx / row_sums_safe
    return mx

def normalize_features_b(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1.).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def pre_social_net(adj, features, labels):
    features = csc_matrix(features.astype(np.uint8))
    labels = labels.astype(np.int32)
    return adj, features.astype(np.float32), labels

def load_pyg_data(file, label_rate, is_blog=None, shuffle=True):
    # 转换特征矩阵为稠密格式并标准化
    # file = 'data/{}.mat'.format(file)
    net = sio.loadmat(file)
    x, a, y = net['attrb'], net['network'], net['group']
    if is_blog:
        a, x, y = pre_social_net(a, x, y)
    if not isinstance(x, scipy.sparse.lil_matrix):
        x = lil_matrix(x)

    x = normalize_features(x)
    x = torch.tensor(x, dtype=torch.float32)
    # print(a)
    # print(type(a))
    A_k = AggTranProbMat(a, 3)
    PPMI_ = ComputePPMI(A_k)
    n_PPMI_ = MyScaleSimMat(PPMI_)  # row normalized PPMI
    n_PPMI_mx = lil_matrix(n_PPMI_)
    a_ppmi = sparse_mx_to_torch_sparse_tensor(n_PPMI_mx)
    num_nodes = x.shape[0]
    # 将稀疏邻接矩阵转换为edge_index格式
    if sp.issparse(a):
        a = a.tocoo()
    edge_index = torch.tensor([a.row, a.col], dtype=torch.long)

    # 处理标签
    y = torch.tensor(y, dtype=torch.float)
    # if y.dim() > 1 and y.size(1) == 1:
    #     y = y.squeeze(1)
    labels = torch.argmax(y, dim=1)
    idx = np.arange(labels.shape[0])
    no_class = int(labels.max() + 1)
    if label_rate < 1:
        tr_size = round((label_rate * labels.shape[0]) // no_class)
    else:
        tr_size = label_rate
    train_size = [tr_size for _ in range(no_class)]

    if shuffle:
        np.random.shuffle(idx)

    idx_train = []
    count = [0 for _ in range(no_class)]
    label_each_class = train_size
    next = 0
    for i in idx:
        if all(count[j] >= label_each_class[j] for j in range(no_class)):
            break
        next += 1
        for j in range(no_class):
            if labels[i] == j and count[j] < label_each_class[j]:
                idx_train.append(i)
                count[j] += 1

    # validation_size =
    validation_size = (labels.shape[0] - tr_size * no_class) // 10
    idx_val = idx[tr_size * no_class:tr_size * no_class + validation_size]
    assert tr_size * no_class + validation_size < len(idx)
    idx_test = idx[tr_size * no_class + validation_size:]

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    # 构造PyG的Data对象
    data = Data(x=x, edge_index=edge_index, y=labels, num_nodes=num_nodes, train_mask=torch.tensor(train_mask), val_mask=torch.tensor(val_mask), test_mask=torch.tensor(test_mask), num_classes=no_class)

    return data, a_ppmi

def load_pyg_data3(file, label_rate, is_blog=None, shuffle=True):
    # 转换特征矩阵为稠密格式并标准化
    # file = 'data/{}.mat'.format(file)
    net = sio.loadmat(file)
    x, a, y = net['attrb'], net['network'], net['group']
    if is_blog:
        a, x, y = pre_social_net(a, x, y)
    if not isinstance(x, scipy.sparse.lil_matrix):
        x = lil_matrix(x)

    x = normalize_features(x)
    x = torch.tensor(x, dtype=torch.float32)
    # print(a)
    # print(type(a))
    # A_k = AggTranProbMat(a, 3)
    # PPMI_ = ComputePPMI(A_k)
    # n_PPMI_ = MyScaleSimMat(PPMI_)  # row normalized PPMI
    # n_PPMI_mx = lil_matrix(n_PPMI_)
    # a_ppmi = sparse_mx_to_torch_sparse_tensor(n_PPMI_mx)
    num_nodes = x.shape[0]
    # 将稀疏邻接矩阵转换为edge_index格式
    if sp.issparse(a):
        a = a.tocoo()
    edge_index = torch.tensor([a.row, a.col], dtype=torch.long)

    # 处理标签
    y = torch.tensor(y, dtype=torch.float)
    # if y.dim() > 1 and y.size(1) == 1:
    #     y = y.squeeze(1)
    labels = torch.argmax(y, dim=1)
    idx = np.arange(labels.shape[0])
    no_class = int(labels.max() + 1)
    if label_rate < 1:
        tr_size = round((label_rate * labels.shape[0]) // no_class)
    else:
        tr_size = label_rate
    train_size = [tr_size for _ in range(no_class)]

    if shuffle:
        np.random.shuffle(idx)

    idx_train = []
    count = [0 for _ in range(no_class)]
    label_each_class = train_size
    next = 0
    for i in idx:
        if all(count[j] >= label_each_class[j] for j in range(no_class)):
            break
        next += 1
        for j in range(no_class):
            if labels[i] == j and count[j] < label_each_class[j]:
                idx_train.append(i)
                count[j] += 1

    # validation_size =
    validation_size = (labels.shape[0] - tr_size * no_class) // 10
    idx_val = idx[tr_size * no_class:tr_size * no_class + validation_size]
    assert tr_size * no_class + validation_size < len(idx)
    idx_test = idx[tr_size * no_class + validation_size:]

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    # 构造PyG的Data对象
    data = Data(x=x, edge_index=edge_index, y=labels, num_nodes=num_nodes, train_mask=torch.tensor(train_mask), val_mask=torch.tensor(val_mask), test_mask=torch.tensor(test_mask), num_classes=no_class)

    return data

import random

def feature_perturbation(feature, noise_factor=0.1):
    """
    对节点特征进行扰动
    :param data: 输入的图数据对象
    :param noise_factor: 噪声因子，控制噪声的强度
    :return: 扰动后的图数据对象
    """
    noise = noise_factor * (torch.randn_like(feature))  # 生成噪声
    feature = feature + noise  # 将噪声添加到节点特征
    return feature


# 2. 结构扰动 (Structural Perturbation)
def structure_perturbation(edge_index, edge_dropout_rate=0.1):
    """
    对图的结构进行扰动（即删除一些边）
    :param data: 输入的图数据对象
    :param edge_dropout_rate: 边删除的比例
    :return: 扰动后的图数据对象
    """
    # 随机丢弃一定比例的边
    num_edges = edge_index.shape[1]
    num_drop = int(edge_dropout_rate * num_edges)

    # 随机选择要删除的边
    drop_indices = random.sample(range(num_edges), num_drop)

    # 删除选中的边
    edge_index = edge_index[:, [i for i in range(num_edges) if i not in drop_indices]]
    return edge_index

def load_pyg_data_2(file, label_rate, per_type, ratio, is_blog=None, shuffle=True):
    # 转换特征矩阵为稠密格式并标准化
    net = sio.loadmat(file)
    x, a, y = net['attrb'], net['network'], net['group']
    if is_blog:
        a, x, y = pre_social_net(a, x, y)
    if not isinstance(x, scipy.sparse.lil_matrix):
        x = lil_matrix(x)

    x = normalize_features(x)
    x = torch.tensor(x, dtype=torch.float32)
    # print(a)
    # print(type(a))
    # A_k = AggTranProbMat(a, 3)
    # PPMI_ = ComputePPMI(A_k)
    # n_PPMI_ = MyScaleSimMat(PPMI_)  # row normalized PPMI
    # n_PPMI_mx = lil_matrix(n_PPMI_)
    a_ppmi = None
    num_nodes = x.shape[0]
    # 将稀疏邻接矩阵转换为edge_index格式
    if sp.issparse(a):
        a = a.tocoo()
    edge_index = torch.tensor([a.row, a.col], dtype=torch.long)

    if per_type == 'feature':
        x = feature_perturbation(x, ratio)
        edge_index = edge_index
    elif per_type == 'structure':
        x = x
    # x = feature_perturbation(x, ratio)
        edge_index = structure_perturbation(edge_index, ratio)
    else:
        x = feature_perturbation(x, ratio)
        edge_index = structure_perturbation(edge_index, ratio)
    # 处理标签
    y = torch.tensor(y, dtype=torch.float)
    # if y.dim() > 1 and y.size(1) == 1:
    #     y = y.squeeze(1)
    labels = torch.argmax(y, dim=1)
    idx = np.arange(labels.shape[0])
    no_class = int(labels.max() + 1)
    if label_rate < 1:
        tr_size = round((label_rate * labels.shape[0]) // no_class)
    else:
        tr_size = label_rate
    train_size = [tr_size for _ in range(no_class)]

    if shuffle:
        np.random.shuffle(idx)

    idx_train = []
    count = [0 for _ in range(no_class)]
    label_each_class = train_size
    next = 0
    for i in idx:
        if all(count[j] >= label_each_class[j] for j in range(no_class)):
            break
        next += 1
        for j in range(no_class):
            if labels[i] == j and count[j] < label_each_class[j]:
                idx_train.append(i)
                count[j] += 1

    # validation_size =
    validation_size = (labels.shape[0] - tr_size * no_class) // 10
    idx_val = idx[tr_size * no_class:tr_size * no_class + validation_size]
    assert tr_size * no_class + validation_size < len(idx)
    idx_test = idx[tr_size * no_class + validation_size:]

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    # 构造PyG的Data对象
    data = Data(x=x, edge_index=edge_index, y=labels, num_nodes=num_nodes, train_mask=torch.tensor(train_mask), val_mask=torch.tensor(val_mask), test_mask=torch.tensor(test_mask), num_classes=no_class)

    return data, a_ppmi


def target_split(data, device):
    num_nodes = data.num_nodes
    idx = np.arange(num_nodes)
    np.random.shuffle(idx)

    # val_ratio = 0.001
    val_size = 6

    idx_val = idx[:val_size]
    idx_test = idx[val_size:]

    val_mask = torch.tensor(sample_mask(idx_val, num_nodes))
    test_mask = torch.tensor(sample_mask(idx_test, num_nodes))
    data.val_mask = val_mask.to(device)
    data.test_mask = test_mask.to(device)
    return data

def target_split_b(data, device):
    num_nodes = data.num_nodes
    idx = np.arange(num_nodes)
    np.random.shuffle(idx)

    # val_ratio = 0.001
    val_size = 6

    idx_val = idx[:val_size]
    idx_test = idx[val_size:]

    val_mask = torch.tensor(sample_mask(idx_val, num_nodes))
    test_mask = torch.tensor(sample_mask(idx_test, num_nodes))
    data.val_mask = val_mask.to(device)
    data.test_mask = test_mask.to(device)
    return data

def load_pyg_data_b(file, label_rate, is_blog=None, shuffle=True):
    # 转换特征矩阵为稠密格式并标准化
    net = sio.loadmat(file)
    x, a, y = net['attrb'], net['network'], net['group']
    if is_blog:
        a, x, y = pre_social_net(a, x, y)
    if not isinstance(x, scipy.sparse.lil_matrix):
        x = lil_matrix(x)

    x = normalize_features_b(x)
    # x = x.astype(np.float32)
    # print(x)
    x = torch.FloatTensor(np.array(x.todense()))
    A_k = AggTranProbMat(a, 3)
    PPMI_ = ComputePPMI(A_k)
    n_PPMI_ = MyScaleSimMat(PPMI_)  # row normalized PPMI
    n_PPMI_mx = lil_matrix(n_PPMI_)
    a_ppmi = sparse_mx_to_torch_sparse_tensor(n_PPMI_mx)
    num_nodes = x.shape[0]
    # 将稀疏邻接矩阵转换为edge_index格式
    if sp.issparse(a):
        a = a.tocoo()
    edge_index = torch.tensor([a.row, a.col], dtype=torch.long)

    # 处理标签
    y = torch.tensor(y, dtype=torch.float)
    # if y.dim() > 1 and y.size(1) == 1:
    #     y = y.squeeze(1)
    labels = torch.argmax(y, dim=1)
    idx = np.arange(labels.shape[0])
    no_class = int(labels.max() + 1)
    if label_rate < 1:
        tr_size = round((label_rate * labels.shape[0]) // no_class)
    else:
        tr_size = label_rate
    train_size = [tr_size for _ in range(no_class)]

    if shuffle:
        np.random.shuffle(idx)

    idx_train = []
    count = [0 for _ in range(no_class)]
    label_each_class = train_size
    next = 0
    for i in idx:
        if all(count[j] >= label_each_class[j] for j in range(no_class)):
            break
        next += 1
        for j in range(no_class):
            if labels[i] == j and count[j] < label_each_class[j]:
                idx_train.append(i)
                count[j] += 1

    # validation_size =
    validation_size = (labels.shape[0] - tr_size * no_class) // 10
    idx_val = idx[tr_size * no_class:tr_size * no_class + validation_size]
    assert tr_size * no_class + validation_size < len(idx)
    idx_test = idx[tr_size * no_class + validation_size:]

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    # 构造PyG的Data对象
    data = Data(x=x, edge_index=edge_index, y=labels, num_nodes=num_nodes, train_mask=torch.tensor(train_mask), val_mask=torch.tensor(val_mask), test_mask=torch.tensor(test_mask), num_classes=no_class)

    return data, a_ppmi


