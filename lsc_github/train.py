from model import SparseGAT
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch_geometric.utils import from_scipy_sparse_matrix
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
import scipy.sparse as sp
import tqdm


class SoftPrecisionLoss(torch.nn.Module):
    """ 自定义精确率损失函数 """
    def __init__(self):
        super(SoftPrecisionLoss, self).__init__()

    def forward(self, logits, labels):
        probas = torch.sigmoid(logits)
        labels = labels.float()

        TP = (probas * labels).sum()
        FP = (probas * (1 - labels)).sum()

        soft_precision = TP / (TP + FP + 1e-8)  # 精确率的软近似
        return 1 - soft_precision  # 最大化精确率 → 最小化此损失


class SoftF1Loss(torch.nn.Module):
    def __init__(self):
        super(SoftF1Loss, self).__init__()

    def forward(self, logits, labels):
        probas = torch.sigmoid(logits)
        labels = labels.float()

        TP = (probas * labels).sum()
        FP = (probas * (1 - labels)).sum()
        FN = ((1 - probas) * labels).sum()

        soft_f1 = 2 * TP / (2 * TP + FP + FN + 1e-8)
        return 1 - soft_f1


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # 模型预测目标类别的概率

        # 正/负样本的 alpha 加权
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # 调制因子，用于聚焦困难样本
        modulating_factor = (1 - pt) ** self.gamma

        focal_loss = alpha_factor * modulating_factor * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train(features_by_cluster=None, adjs_by_cluster=None, labels_by_cluster=None, features=None, adjs=None, label=None):
    base_path = 'E:/Project/traffic/lsc/model/'
    # 模型参数配置
    hidden_dim = 16  # 隐藏层维度
    dropout = 0.2  # 随机失活率
    alpha = 0.2  # LeakyReLU负斜率
    nheads = 8  # GAT注意力头数
    lr = 5e-4
    weight_decay = 5e-4
    epochs = 10000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 处理多类别司机数据
    if features_by_cluster is not None and adjs_by_cluster is not None and labels_by_cluster is not None:
        print('==> Preparing multi-cluster driver training dataset..')
        
        # 为每个类别准备数据
        cluster_data = {}
        total_features_dim = 782
        
        for cluster_id in features_by_cluster.keys():
            features = features_by_cluster[cluster_id]
            adjs = adjs_by_cluster[cluster_id]
            labels = labels_by_cluster[cluster_id]
            
            N = len(features)
            
            # 检查特征维度一致性
            for i in range(len(features)):
                if len(features[i]) < total_features_dim:
                    features[i].extend([0] * (total_features_dim - len(features[i])))
                elif len(features[i]) > total_features_dim:
                    features[i] = features[i][:total_features_dim]
            
            # 将字典格式的邻接表转换为稀疏矩阵
            if isinstance(adjs, dict):
                # 创建邻接矩阵
                adj_matrix = np.zeros((N, N), dtype=np.float32)
                for u, neighbors in adjs.items():
                    for v, weight in neighbors.items():
                        if u < N and v < N:  # 确保索引在范围内
                            adj_matrix[u, v] = weight
                adj = csr_matrix(adj_matrix)
            else:
                adj = csr_matrix(adjs)
                
            features_tensor = torch.tensor(features, dtype=torch.float)
            labels_tensor = torch.tensor(labels, dtype=torch.float)
            
            # 确保所有类别的特征维度一致
            if total_features_dim is None:
                total_features_dim = features_tensor.shape[1]
            else:
                assert features_tensor.shape[1] == total_features_dim, f"Feature dimensions must be consistent across clusters"
            
            # 划分训练/验证/测试集
            labels_array = np.array(labels)
            train_idx, test_idx = train_test_split(np.arange(N), test_size=0.2, stratify=labels_array)
            train_idx, val_idx = train_test_split(train_idx, test_size=0.25, stratify=labels_array[train_idx])
            
            train_mask = torch.zeros(N, dtype=torch.bool)
            val_mask = torch.zeros(N, dtype=torch.bool)
            test_mask = torch.zeros(N, dtype=torch.bool)
            train_mask[train_idx] = True
            val_mask[val_idx] = True
            test_mask[test_idx] = True
            
            edge_index, edge_weight = from_scipy_sparse_matrix(adj)
            
            data = Data(x=features_tensor, edge_index=edge_index, y=labels_tensor,
                       train_mask=train_mask, val_mask=val_mask, test_mask=test_mask).to(device)
            
            # 计算正样本权重
            y_train = data.y[data.train_mask].float()
            num_pos = y_train.sum().item()
            num_neg = len(y_train) - num_pos
            pos_weight = torch.tensor([num_neg / (num_pos + 1e-5)]).to(device)
            
            cluster_data[cluster_id] = {
                'data': data,
                'pos_weight': pos_weight,
                'train_idx': train_idx,
                'val_idx': val_idx,
                'test_idx': test_idx
            }
            
            print(f'Cluster {cluster_id}: {N} nodes, {num_pos} positive samples, pos_weight: {pos_weight[0].cpu():.4f}')
        
        # 初始化模型与优化器
        model = SparseGAT(
            nfeat=total_features_dim,
            nhid=hidden_dim,
            nclass=1,
            dropout=dropout,
            alpha=alpha,
            nheads=nheads
        ).to(device)
        
        # # 尝试加载预训练模型
        try:
            model.load_state_dict(torch.load('./model/best_model_multi_cluster.pth', map_location=device))
            print("Loaded pre-trained model")
        except:
            print("No pre-trained model found, starting from scratch")
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        best_val_f1 = 0.0
        best_model_path = base_path + "best_model_multi_cluster.pth"
        
        print('Training multi-cluster model')
        # 训练循环
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 计算所有类别的总损失
            total_loss = 0.0
            cluster_losses = {}
            
            for cluster_id, cluster_info in cluster_data.items():
                data = cluster_info['data']
                pos_weight = cluster_info['pos_weight']
                
                out = model(data.x, data.edge_index)
                
                # 计算各种损失
                focal_criterion = FocalLoss(alpha=pos_weight.item() / (pos_weight.item() + 1), gamma=2)
                f1_criterion = SoftF1Loss()
                precision_criterion = SoftPrecisionLoss()
                
                loss_focal = focal_criterion(out[data.train_mask], data.y[data.train_mask].float())
                loss_f1 = f1_criterion(out[data.train_mask], data.y[data.train_mask].float())
                loss_precision = precision_criterion(out[data.train_mask], data.y[data.train_mask].float())
                
                # 当前类别的损失
                cluster_loss = 0.5 * loss_focal + 0.5 * loss_f1
                cluster_losses[cluster_id] = cluster_loss.item()
                total_loss += cluster_loss
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            # 验证集评估
            model.eval()
            total_val_f1 = 0.0
            total_val_acc = 0.0
            total_val_precision = 0.0
            total_val_recall = 0.0
            cluster_count = 0
            
            with torch.no_grad():
                for cluster_id, cluster_info in cluster_data.items():
                    data = cluster_info['data']
                    out = model(data.x, data.edge_index)
                    pred = (torch.sigmoid(out) > 0.5).long()
                    
                    val_pred = pred[data.val_mask]
                    val_y = data.y[data.val_mask]
                    
                    if val_y.sum() > 0:  # 确保有正样本
                        f1 = f1_score(val_y.cpu(), val_pred.cpu(), pos_label=1)
                        recall = recall_score(val_y.cpu(), val_pred.cpu(), pos_label=1)
                        acc = accuracy_score(val_y.cpu(), val_pred.cpu())
                        precision = precision_score(val_y.cpu(), val_pred.cpu(), pos_label=1)
                        
                        total_val_f1 += f1
                        total_val_acc += acc
                        total_val_precision += precision
                        total_val_recall += recall
                        cluster_count += 1
            
            # 计算平均指标
            if cluster_count > 0:
                avg_val_f1 = total_val_f1 / cluster_count
                avg_val_acc = total_val_acc / cluster_count
                avg_val_precision = total_val_precision / cluster_count
                avg_val_recall = total_val_recall / cluster_count
                
                print(f"Epoch {epoch + 1:03d}, Total Loss: {total_loss.item():.4f}, "
                      f"Avg Val Acc: {avg_val_acc:.4f}, Avg F1: {avg_val_f1:.4f}, "
                      f"Avg Precision: {avg_val_precision:.4f}, Avg Recall: {avg_val_recall:.4f}")
                
                # 保存最佳模型
                if avg_val_f1 > best_val_f1 and epoch > 10:
                    best_val_f1 = avg_val_f1
                    torch.save(model.state_dict(), best_model_path)
                    print(f"New best model saved with avg F1: {best_val_f1:.4f}")
            
            model.train()
        
        print(f"Multi-cluster training finished. Best validation F1: {best_val_f1:.4f}")
        print(f"Best model saved to {best_model_path}")
        
        # =================== 测试阶段 =====================
        print("Testing multi-cluster model")
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        
        # 测试不同阈值
        for threshold_offset in range(0, 10):
            threshold = 0.5 + 0.05 * threshold_offset
            total_test_f1 = 0.0
            total_test_acc = 0.0
            total_test_precision = 0.0
            total_test_recall = 0.0
            cluster_count = 0
            
            with torch.no_grad():
                for cluster_id, cluster_info in cluster_data.items():
                    data = cluster_info['data']
                    out = model(data.x, data.edge_index)
                    pred = (torch.sigmoid(out) > threshold).long()
                    
                    test_pred = pred[data.test_mask]
                    test_y = data.y[data.test_mask]
                    
                    if test_y.sum() > 0:
                        test_f1 = f1_score(test_y.cpu(), test_pred.cpu(), pos_label=1)
                        test_recall = recall_score(test_y.cpu(), test_pred.cpu(), pos_label=1)
                        test_precision = precision_score(test_y.cpu(), test_pred.cpu(), pos_label=1)
                        test_acc = accuracy_score(test_y.cpu(), test_pred.cpu())
                        
                        total_test_f1 += test_f1
                        total_test_acc += test_acc
                        total_test_precision += test_precision
                        total_test_recall += test_recall
                        cluster_count += 1
            
            if cluster_count > 0:
                avg_test_f1 = total_test_f1 / cluster_count
                avg_test_acc = total_test_acc / cluster_count
                avg_test_precision = total_test_precision / cluster_count
                avg_test_recall = total_test_recall / cluster_count
                
                print(f"Threshold: {threshold:.4f}, Avg Test Acc: {avg_test_acc:.4f}, "
                      f"Avg F1: {avg_test_f1:.4f}, Avg Precision: {avg_test_precision:.4f}, "
                      f"Avg Recall: {avg_test_recall:.4f}")
        
        return model, cluster_data
    
    # 原有的单图训练逻辑（保持向后兼容）
    else:
        print('==> Preparing single graph training dataset..')
        if features is None:
            adj = sp.load_npz(base_path + 'adj_matrix.npz')
            features = torch.load(base_path + 'features.pt')
            N = len(features)
            label = torch.load(base_path + 'labels.pt')
        else:
            N = len(features)
            adj = csr_matrix(adjs)
            features = torch.tensor(features, dtype=torch.float)
            label = torch.tensor(label)
            sp.save_npz(base_path + 'adj_matrix.npz', adj)
            torch.save(features, base_path + 'features.pt')
            torch.save(label, base_path + 'labels.pt')

        edge_index, edge_weight = from_scipy_sparse_matrix(adj)
        train_idx, test_idx = train_test_split(np.arange(N), test_size=0.2, stratify=label)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.25, stratify=label[train_idx])

        train_mask = torch.zeros(N, dtype=torch.bool)
        val_mask = torch.zeros(N, dtype=torch.bool)
        test_mask = torch.zeros(N, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        np.save(base_path + "train_idx.npy", train_idx)
        np.save(base_path + "val_idx.npy", val_idx)
        np.save(base_path + "test_idx.npy", test_idx)

        data = Data(x=features, edge_index=edge_index, y=label,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask).to(device)

        y_train = data.y[data.train_mask].float()
        num_pos = y_train.sum().item()  # 正样本总数
        num_neg = len(y_train) - num_pos  # 负样本总数
        pos_weight = torch.tensor([num_neg / (num_pos + 1e-5)]).to(device)
        print(f'Pos weight: {pos_weight[0].cpu():.4f}')

        # 初始化模型与优化器
        model = SparseGAT(
            nfeat=features.shape[1],
            nhid=hidden_dim,
            nclass=1,
            dropout=dropout,
            alpha=alpha,
            nheads=nheads
        ).to(device)
        model.load_state_dict(torch.load('./model/dim-16-f1-0.52-p-0.44-r-0.63/best_model_new.pth', map_location=device))
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_f1 = 0.0
        best_model_path = base_path + "best_model_new.pth"

        print('Training')
        # 训练循环
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)

            focal_criterion = FocalLoss(alpha=pos_weight.item() / (pos_weight.item() + 1), gamma=2)
            f1_criterion = SoftF1Loss()
            precision_criterion = SoftPrecisionLoss()
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            loss_focal = focal_criterion(out[data.train_mask], data.y[data.train_mask].float())
            loss_f1 = f1_criterion(out[data.train_mask], data.y[data.train_mask].float())
            loss_precision = precision_criterion(out[data.train_mask], data.y[data.train_mask].float())

            loss = 0.0 * loss_focal + 1.0 * loss_f1
            # loss = criterion(out[data.train_mask].squeeze(), data.y[data.train_mask].float())
            loss.backward()
            optimizer.step()

            # 验证集准确率
            model.eval()
            with torch.no_grad():
                pred = (torch.sigmoid(out) > 0.5).long()
                val_pred = pred[data.val_mask]
                val_y = data.y[data.val_mask]

                f1 = f1_score(val_y.cpu(), val_pred.cpu(), pos_label=1)
                recall = recall_score(val_y.cpu(), val_pred.cpu(), pos_label=1)
                acc = accuracy_score(val_y.cpu(), val_pred.cpu())
                precision = precision_score(val_y.cpu(), val_pred.cpu(), pos_label=1)

            print(f"Epoch {epoch + 1:03d}, Loss: {loss.item():.4f},"
                  f" Val Acc: {acc:.4f}, F1 Score: {f1:.4f}, Test Precision: {precision:.4f}, Recall: {recall:.4f}")

            if f1 > best_val_f1 and epoch > 10:
                best_val_f1 = f1
                torch.save(model.state_dict(), best_model_path)

        print(f"Training finished. Best validation f1: {best_val_f1:.4f}")
        print(f"Best model saved to {best_model_path}")

        # =================== 测试阶段 =====================
        print("Testing")
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        out = model(data.x, data.edge_index)
        for i in range(0, 10):
            pred = (torch.sigmoid(out) > 0.5 + 0.05 * i).long()

            test_pred = pred[data.test_mask]
            test_y = data.y[data.test_mask]
            test_f1 = f1_score(test_y.cpu(), test_pred.cpu(), pos_label=1)
            test_recall = recall_score(test_y.cpu(), test_pred.cpu(), pos_label=1)
            test_precision = precision_score(test_y.cpu(), test_pred.cpu(), pos_label=1)
            test_acc = accuracy_score(test_y.cpu(), test_pred.cpu())

            print(f"Thresholding: {0.5 + 0.05 * i:.4f}, Test Accuracy: {test_acc:.4f}, F1 Score: {test_f1:.4f},"
                  f" Test Precision: {test_precision:.4f},  Recall: {test_recall:.4f}")
        
        return model, data


def test(features, adjs, label, model_path="./model/best_model.pth"):
    hidden_dim = 128
    dropout = 0.4
    alpha = 0.2
    nheads = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 构建图数据
    N = len(features)
    adj = csr_matrix(adjs)
    features = torch.tensor(features, dtype=torch.float)
    label = torch.tensor(label)
    edge_index, edge_weight = from_scipy_sparse_matrix(adj)

    test_idx = np.load("./model/test_idx.npy")
    test_mask = torch.zeros(N, dtype=torch.bool)
    test_mask[test_idx] = True

    data = Data(x=features, edge_index=edge_index, y=label, test_mask=test_mask).to(device)

    # 初始化模型并加载参数
    model = SparseGAT(
        nfeat=features.shape[1],
        nhid=hidden_dim,
        nclass=1,
        dropout=dropout,
        alpha=alpha,
        nheads=nheads
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 推理
    out = model(data.x, data.edge_index)
    pred = (torch.sigmoid(out) > 0.5).long()
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
    acc = correct / data.test_mask.sum().item()
    print(f"Test Accuracy: {acc:.4f}")

    from sklearn.metrics import classification_report
    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = pred[data.test_mask].cpu().numpy()
    print(classification_report(y_true, y_pred, digits=4))


def test_multi_cluster(features_by_cluster, adjs_by_cluster, labels_by_cluster, model_path="./model/best_model_multi_cluster.pth"):
    """
    测试多类别司机模型
    
    :param features_by_cluster: 每个类别的特征字典
    :param adjs_by_cluster: 每个类别的邻接矩阵字典
    :param labels_by_cluster: 每个类别的标签字典
    :param model_path: 模型路径
    """
    hidden_dim = 16
    dropout = 0.2
    alpha = 0.2
    nheads = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 为每个类别准备数据
    cluster_data = {}
    total_features_dim = 782
    
    for cluster_id in features_by_cluster.keys():
        features = features_by_cluster[cluster_id]
        adjs = adjs_by_cluster[cluster_id]
        labels = labels_by_cluster[cluster_id]
        
        N = len(features)
        
        # 检查特征维度一致性
        if total_features_dim is None:
            total_features_dim = len(features[0]) if features else 0
            print(f'Setting feature dimension to: {total_features_dim}')
        else:
            current_dim = len(features[0]) if features else 0
            if current_dim != total_features_dim or True:
                print(f'Cluster {cluster_id}: feature dimension mismatch ({current_dim} vs {total_features_dim}), padding...')
                # 对每个节点的特征进行填充或截断
                for i in range(len(features)):
                    if len(features[i]) < total_features_dim:
                        features[i].extend([0] * (total_features_dim - len(features[i])))
                    elif len(features[i]) > total_features_dim:
                        features[i] = features[i][:total_features_dim]
        
        # 将字典格式的邻接表转换为稀疏矩阵
        if isinstance(adjs, dict):
            # 创建邻接矩阵
            adj_matrix = np.zeros((N, N), dtype=np.float32)
            for u, neighbors in adjs.items():
                for v, weight in neighbors.items():
                    if u < N and v < N:  # 确保索引在范围内
                        adj_matrix[u, v] = weight
            adj = csr_matrix(adj_matrix)
        else:
            adj = csr_matrix(adjs)
            
        features_tensor = torch.tensor(features, dtype=torch.float)
        labels_tensor = torch.tensor(labels, dtype=torch.float)
        
        # 确保所有类别的特征维度一致
        if total_features_dim is None:
            total_features_dim = features_tensor.shape[1]
        else:
            assert features_tensor.shape[1] == total_features_dim, f"Feature dimensions must be consistent across clusters"
        
        # 划分训练/验证/测试集
        labels_array = np.array(labels)
        train_idx, test_idx = train_test_split(np.arange(N), test_size=0.2, stratify=labels_array)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.25, stratify=labels_array[train_idx])
        
        test_mask = torch.zeros(N, dtype=torch.bool)
        test_mask[test_idx] = True
        
        edge_index, edge_weight = from_scipy_sparse_matrix(adj)
        
        data = Data(x=features_tensor, edge_index=edge_index, y=labels_tensor, test_mask=test_mask).to(device)
        
        cluster_data[cluster_id] = {
            'data': data,
            'test_idx': test_idx
        }
        
        print(f'Cluster {cluster_id}: {N} nodes, test samples: {len(test_idx)}')

    # 初始化模型并加载参数
    model = SparseGAT(
        nfeat=total_features_dim,
        nhid=hidden_dim,
        nclass=1,
        dropout=dropout,
        alpha=alpha,
        nheads=nheads
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    except:
        print(f"Failed to load model from {model_path}")
        return
    
    model.eval()

    # 测试不同阈值
    print("\n=== Multi-Cluster Model Testing Results ===")
    best_threshold = 0.5
    best_avg_f1 = 0.0
    
    for threshold_offset in range(0, 10):
        threshold = 0.5 + 0.05 * threshold_offset
        total_test_f1 = 0.0
        total_test_acc = 0.0
        total_test_precision = 0.0
        total_test_recall = 0.0
        cluster_count = 0
        
        with torch.no_grad():
            for cluster_id, cluster_info in cluster_data.items():
                data = cluster_info['data']
                out = model(data.x, data.edge_index)
                pred = (torch.sigmoid(out) > threshold).long()
                
                test_pred = pred[data.test_mask]
                test_y = data.y[data.test_mask]
                
                if test_y.sum() > 0:
                    test_f1 = f1_score(test_y.cpu(), test_pred.cpu(), pos_label=1)
                    test_recall = recall_score(test_y.cpu(), test_pred.cpu(), pos_label=1)
                    test_precision = precision_score(test_y.cpu(), test_pred.cpu(), pos_label=1)
                    test_acc = accuracy_score(test_y.cpu(), test_pred.cpu())
                    
                    total_test_f1 += test_f1
                    total_test_acc += test_acc
                    total_test_precision += test_precision
                    total_test_recall += test_recall
                    cluster_count += 1
                    
                    print(f"  Cluster {cluster_id}: F1={test_f1:.4f}, Acc={test_acc:.4f}, P={test_precision:.4f}, R={test_recall:.4f}")
        
        if cluster_count > 0:
            avg_test_f1 = total_test_f1 / cluster_count
            avg_test_acc = total_test_acc / cluster_count
            avg_test_precision = total_test_precision / cluster_count
            avg_test_recall = total_test_recall / cluster_count
            
            print(f"Threshold: {threshold:.4f}, Avg Test Acc: {avg_test_acc:.4f}, "
                  f"Avg F1: {avg_test_f1:.4f}, Avg Precision: {avg_test_precision:.4f}, "
                  f"Avg Recall: {avg_test_recall:.4f}")
            
            if avg_test_f1 > best_avg_f1:
                best_avg_f1 = avg_test_f1
                best_threshold = threshold
    
    print(f"\nBest threshold: {best_threshold:.4f} with average F1: {best_avg_f1:.4f}")
