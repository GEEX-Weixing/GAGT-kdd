import torch.nn.functional as F
import torch
from torch import optim
import torch.nn as nn
from get_data_cross_network import load_pyg_data3, target_split
from model import DD, Style
from utils import for_gae, compute_accuracy_teacher_mask, intra_domain_contrastive_loss, compute_accuracy_teacher, loss_function, set_random_seeds, entropy_loss_f, gamma_re_loss, intra_domain_contrastive_loss
import itertools
from sklearn.metrics import f1_score
import numpy as np
import time
from models import S_Encoder, P_Encoder, Decoder, reparameterize, Generator, Generator_mdt2
from torch_geometric.utils import to_dense_adj
import scipy.sparse as sp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
source_graphs = ['acmv9', 'citationv1', 'dblpv7']
target_graphs = ['acmv9', 'citationv1', 'dblpv7']
for i in range(3):
    for j in range(3):
        if i == j:
            pass
        else:
            source_graph = source_graphs[i]
            target_graph = target_graphs[j]
            if source_graph != target_graphs:
                print("S -----> T: {}------->{}".format(source_graph, target_graph))

                data_s = load_pyg_data3('data/{}.mat'.format(source_graph), label_rate=0.5)
                data_t = load_pyg_data3('data/{}.mat'.format(target_graph), label_rate=0.5)
                # 存储5次运行的结果
                all_best_accs = []
                all_best_ma_f1s = []
                all_best_mi_f1s = []
                for run in range(5):
                    print(f"\n开始第 {run+1} 次运行...")
                    set_random_seeds(42 + run)  # 每次运行使用不同的随机种子
                    neg_sample_generator = torch.Generator(device=device)
                    data_s.adj = sp.coo_matrix(to_dense_adj(data_s.edge_index.detach().cpu()).squeeze(0))
                    data_t.adj = sp.coo_matrix(to_dense_adj(data_t.edge_index.detach().cpu()).squeeze(0))
                    data_s = data_s.to(device)
                    data_t = data_t.to(device)
                    data_t = target_split(data_t, device)
                    adj_label_s, norm_s, pos_weight_s = for_gae(data_s.x, data_s.adj, device)
                    adj_label_t, norm_t, pos_weight_t = for_gae(data_t.x, data_t.adj, device)
                    shared_encoder = S_Encoder(data_s.x.shape[1], 512, 64).to(device)
                    private_encoder = P_Encoder(data_s.x.shape[1], 512, 64).to(device)
                    decoder = Decoder(128, data_s.x.shape[1]).to(device)
                    style_s = Style(64, 0.01, 0.4).to(device)
                    style_t = Style(64, 0.01, 0.4).to(device)
                    generator_s = Generator_mdt2().to(device)
                    generator_t = Generator_mdt2().to(device)
                    discriminator_d = DD(64, 16, 2, 0.6).to(device)
                    cls_model = nn.Sequential(nn.Linear(64, data_s.num_classes),).to(device)
                    optimizer = optim.Adam(itertools.chain(shared_encoder.parameters(), private_encoder.parameters(), decoder.parameters(),
                                                           style_s.parameters(), style_t.parameters(), discriminator_d.parameters(),
                                                           cls_model.parameters()), lr=1e-3, weight_decay=5e-4)
                    cls_loss = nn.CrossEntropyLoss().to(device)
                    domain_loss = nn.CrossEntropyLoss()

                    best_acc = 0
                    best_maf = 0
                    best_mif = 0
                    losssss = []
                    accs = []
                    for epoch in range(300):
                        start_time = time.time()
                        rate = min((epoch+1) / 300, 0.05)

                        mu_s_s, logvar_s_s = shared_encoder(data_s.x, data_s.edge_index)
                        mu_s_p, logvar_s_p = private_encoder(data_s.x)
                        # z_s
                        z_s_s = reparameterize(mu_s_s, logvar_s_s)
                        z_s_p = reparameterize(mu_s_p, logvar_s_p)
                        mu_s = torch.cat((mu_s_s, mu_s_p), 1)
                        logvar_s = torch.cat((logvar_s_s, logvar_s_p), 1)
                        concat_z_s = torch.cat((z_s_s, z_s_p), 1)
                        recon_adj_s, recon_fea_s = decoder(concat_z_s)

                        mu_t_s, logvar_t_s = shared_encoder(data_t.x, data_t.edge_index)
                        mu_t_p, logvar_t_p = private_encoder(data_t.x)
                        # z_t
                        z_t_s = reparameterize(mu_t_s, logvar_t_s)
                        z_t_p = reparameterize(mu_t_p, logvar_t_p)
                        mu_t = torch.cat((mu_t_s, mu_t_p), 1)
                        logvar_t = torch.cat((logvar_t_s, logvar_t_p), 1)
                        concat_z_t = torch.cat((z_t_s, z_t_p), 1)
                        recon_adj_t, recon_fea_t = decoder(concat_z_t)
                        adj_recover_loss_s = loss_function(recon_adj_s, adj_label_s, mu_s, logvar_s, data_s.num_nodes, norm_s, pos_weight_s)
                        adj_recover_loss_t = loss_function(recon_adj_t, adj_label_t, mu_t, logvar_t, data_t.num_nodes, norm_t, pos_weight_t)
                        fea_recover_loss_s = gamma_re_loss(recon_fea_s, data_s.x, gamma=0.01)
                        fea_recover_loss_t = gamma_re_loss(recon_fea_t, data_t.x, gamma=0.01)

                        intra_domain_cl_s = intra_domain_contrastive_loss(data_s.x, z_s_s, z_s_p, decoder, generator=neg_sample_generator, domain_predictor=None)
                        intra_domain_cl_t = intra_domain_contrastive_loss(data_t.x, z_t_s, z_t_p, decoder, generator=neg_sample_generator, domain_predictor=None)

                        mu_s_s_re, logvar_s_s_re = shared_encoder(recon_fea_s, data_s.edge_index)
                        mu_t_s_re, logvar_t_s_re = shared_encoder(recon_fea_t, data_t.edge_index)

                        # \hat{z}_{s/t}
                        z_s_s_rec = reparameterize(mu_s_s_re, logvar_s_s_re)
                        z_t_s_rec = reparameterize(mu_t_s_re, logvar_t_s_re)

                        s_style = style_s(z_s_p)
                        t_style = style_t(z_t_p)
                        z_s_s_trans = generator_s(z_s_s, t_style)
                        z_t_s_trans = generator_t(z_t_s, s_style)

                        domain_output_s = discriminator_d(z_s_s, data_s.edge_index, rate)
                        domain_output_s_g = discriminator_d(z_s_s_trans, data_s.edge_index, rate)
                        domain_output_t = discriminator_d(z_t_s, data_t.edge_index, rate)
                        domain_output_t_g = discriminator_d(z_t_s_trans, data_t.edge_index, rate)
                        err_s_domain = domain_loss(domain_output_s, torch.zeros(domain_output_s.size(0)).type(torch.LongTensor).to(device)) + \
                                       domain_loss(domain_output_s_g, torch.zeros(domain_output_s_g.size(0)).type(torch.LongTensor).to(device))
                        err_t_domain = domain_loss(domain_output_t, torch.ones(domain_output_t.size(0)).type(torch.LongTensor).to(device)) + \
                                       domain_loss(domain_output_t_g, torch.ones(domain_output_t_g.size(0)).type(torch.LongTensor).to(device))

                        grl_loss = err_s_domain + err_t_domain
                        logit_s = cls_model(z_s_s)
                        cls_loss_s_o = cls_loss(logit_s, data_s.y)

                        logit_t = cls_model(z_t_s)
                        cls_loss_t_o = cls_loss(logit_t[data_t.val_mask], data_t.y[data_t.val_mask])
                        cls_loss_s = cls_loss_s_o + cls_loss_t_o
                        entropy_loss = entropy_loss_f(logit_t)

                        acc_s = compute_accuracy_teacher(torch.argmax(logit_s.detach(), dim=1), data_s.y)
                        acc_test = compute_accuracy_teacher_mask(torch.argmax(logit_t.detach(), dim=1), data_t.y, data_t.test_mask)
                        mif1_test = f1_score(data_t.y.cpu().numpy(), torch.argmax(logit_t.detach(), dim=1).cpu().numpy(), average='micro')
                        maf1_test = f1_score(data_t.y.cpu().numpy(), torch.argmax(logit_t.detach(), dim=1).cpu().numpy(), average='macro')
                        loss = entropy_loss * (epoch / 300) + adj_recover_loss_s + adj_recover_loss_t + fea_recover_loss_s + fea_recover_loss_t + intra_domain_cl_s + intra_domain_cl_t + grl_loss + cls_loss_s
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        losssss.append(loss.item())
                        accs.append(acc_test)
                        if acc_test > best_acc:
                            best_acc = acc_test
                        if maf1_test > best_maf:
                            best_maf = maf1_test
                        if mif1_test > best_mif:
                            best_mif = mif1_test
                        if epoch % 10 == 0:
                            print(
                                "Epoch: [{}/{}] | En Loss: {:.4f} | A_R_S Loss: {:.4f} | A_R_T Loss: {:.4f} | F_R_S Loss: {:.4f} | F_R_T Loss: {:.4f} | CL_S Loss: {:.4f} | CL_T Loss: {:.4f} | GRL Loss: {:.4f} | CLS Loss: {:.4f} | Train acc: {:.4f} | Test acc_real: {:.4f}  | Test acc: {:.4f} | MiF1: {:.4f} | MaF1: {:.4f}".format(
                                    epoch + 1, 300, entropy_loss.item(), adj_recover_loss_s.item(), adj_recover_loss_t.item(), fea_recover_loss_s.item(),
                                    fea_recover_loss_t.item(), intra_domain_cl_s.item(), intra_domain_cl_t.item(), grl_loss.item(), cls_loss_s.item(), acc_s, acc_test, best_acc, best_mif, best_maf))

                    # 记录本次运行的最佳结果
                    all_best_accs.append(best_acc)
                    all_best_ma_f1s.append(best_maf)
                    all_best_mi_f1s.append(best_mif)
                # 计算平均值和标准差
                mean_acc = np.mean(all_best_accs)
                std_acc = np.std(all_best_accs)
                mean_maf1 = np.mean(all_best_ma_f1s)
                std_maf1 = np.std(all_best_ma_f1s)
                mean_mif1 = np.mean(all_best_mi_f1s)
                std_mif1 = np.std(all_best_mi_f1s)

                print(f"Test Accuracy - Mean: {mean_acc:.4f} ± {std_acc:.4f}")
                print(f"Macro F1 - Mean: {mean_maf1:.4f} ± {std_maf1:.4f}")
                print(f"Micro F1 - Mean: {mean_mif1:.4f} ± {std_mif1:.4f}")

















