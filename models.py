import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class P_Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super().__init__()
        self.conv1 = nn.Linear(in_channels, hidden_dim)
        self.conv_mu = nn.Linear(hidden_dim, latent_dim)
        self.conv_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.dropout(h, p=0.3, training=self.training)
        mu = self.conv_mu(h)
        logvar = self.conv_logvar(h)
        return mu, logvar

class S_Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim, heads=2):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_dim, heads=heads, concat=True)
        conv1_out_dim = hidden_dim * heads
        self.conv_mu = GATConv(conv1_out_dim, latent_dim, heads=1, concat=False)
        self.conv_logvar = GATConv(conv1_out_dim, latent_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        h = F.elu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.3, training=self.training)
        mu = self.conv_mu(h, edge_index)
        logvar = self.conv_logvar(h, edge_index)
        return mu, logvar


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, feat_dim):
        super().__init__()
        self.feat_decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, latent_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(latent_dim * 2, feat_dim)
        )

    def forward(self, z):
        x_recon = self.feat_decoder(z)
        adj_recon = torch.mm(z, z.t())  # 内积重建邻接
        return adj_recon, x_recon


class AdaIN(nn.Module):
    def __init__(self, input_dim, style_dim):
        super(AdaIN, self).__init__()
        self.style_fc1 = nn.Linear(style_dim, input_dim)
        self.style_fc2 = nn.Linear(style_dim, input_dim)

    def forward(self, x, style):
        # x: [B, C, N]
        # style: [B, style_dim]

        mean = torch.mean(x, dim=2, keepdim=True)   # [B, C, 1]
        std = torch.std(x, dim=2, keepdim=True)     # [B, C, 1]

        gamma = self.style_fc1(style).unsqueeze(-1)  # [B, C, 1]
        beta = self.style_fc2(style).unsqueeze(-1)   # [B, C, 1]

        x_normalized = (x - mean) / (std + 1e-8)
        out = gamma * x_normalized + beta
        return out

class Generator_mdt(torch.nn.Module):
    def __init__(self, input_num, input_dim, hidden_dim, out_num, dropout):
        super(Generator_mdt, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.trs1 = nn.Linear(input_num, out_num)
        self.trs2 = nn.Linear(out_num, input_num)
        # self.fc2 = GATConv(hidden_dim, input_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(out_num)
        self.bn3 = nn.BatchNorm1d(input_dim)
        self.bn4 = nn.BatchNorm1d(input_num)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.adain1 = AdaIN(64, 64)
        self.adain2 = AdaIN(256, 64)
        self.dropout = dropout

    def forward(self, x, style):
        x = x.T
        x = self.act(self.bn2(self.trs1(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.T
        x = self.adain1(x, style)
        x = self.act(self.bn1(self.fc1(x)))
        x = self.adain2(x, style)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.act(self.bn3(self.fc2(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.T
        x = self.act(self.bn4(self.trs2(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.T
        return x

class Generator_mdt2(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, style_dim=64, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.adain1 = AdaIN(hidden_dim, style_dim)
        self.adain2 = AdaIN(input_dim, style_dim)
        self.dropout = dropout
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, style):
        # x: [N, D] or [B, N, D]  --> 我们支持 [N, D] 自动加 batch dim
        # style: [S,] or [B, S]   --> 支持 [S,] 自动加 batch dim

        if x.dim() == 2:
            x = x.unsqueeze(0)          # [N, D] → [1, N, D]
        if style.dim() == 1:
            style = style.unsqueeze(0)  # [S] → [1, S]

        B, N, D = x.shape
        assert style.shape[0] == B, f"Batch mismatch: x has {B}, style has {style.shape[0]}"

        # First MLP + AdaIN
        x = self.act(self.fc1(x))                     # [B, N, hidden_dim]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(1, 2)                         # [B, hidden_dim, N]
        x = self.adain1(x, style)                     # AdaIN over feature dim
        x = x.transpose(1, 2)                         # [B, N, hidden_dim]

        # Second MLP + AdaIN
        x = self.act(self.fc2(x))                     # [B, N, input_dim]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(1, 2)                         # [B, input_dim, N]
        x = self.adain2(x, style)
        x = x.transpose(1, 2)                         # [B, N, input_dim]

        return x.squeeze(0) if B == 1 else x         # 保持输入输出维度一致

class Generator(torch.nn.Module):
    def __init__(self, input_num, input_dim, hidden_dim, out_num, dropout):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.trs = nn.Linear(input_num, out_num)
        self.fc2 = GATConv(hidden_dim, input_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(out_num)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.adain1 = AdaIN(64, 64)
        self.adain2 = AdaIN(256, 64)
        self.adain3 = AdaIN(64, 64)
        self.dropout = dropout

    def forward(self, x, edge_index, style):
        x = x.T
        x = self.act(self.bn2(self.trs(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.T
        x = self.adain1(x, style)
        x = self.act(self.bn1(self.fc1(x)))
        x = self.adain2(x, style)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.act(self.fc2(x, edge_index))
        x = self.adain3(x, style)
        return x


# ----------------------------
# Gradient Reversal Layer (GRL)
# ----------------------------
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

# ----------------------------
# GAT-based Domain Classifier (for 4 domains)
# ----------------------------
class DomainClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_heads=4, num_domains=2):
        super().__init__()
        self.grl = GradientReversalLayer(lambda_=1.0)

        # 第一层 GAT
        self.gat1 = GATConv(in_dim, hidden_dim, heads=num_heads, dropout=0.6)
        # 第二层 GAT：输出维度为 hidden_dim（concat=False）
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=0.6)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_domains)
        )

    def forward(self, x, edge_index, batch=None):
        """
        x: [N, in_dim] – 节点特征（已包含语义+风格）
        edge_index: [2, E]
        batch: [N,] – 图批次指示向量（若单图可为 None）
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.grl(x)  # 在输入处应用 GRL（也可在 GAT 后，但通常在输入）

        # GAT layers
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_index)

        # Global mean pooling to get graph-level representation
        graph_emb = global_mean_pool(x, batch)  # [num_graphs, hidden_dim]

        logits = self.classifier(graph_emb)  # [num_graphs, 4]
        return logits









