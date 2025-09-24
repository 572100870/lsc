import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, ndcg_score
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import csr_matrix
from model import SparseGAT
import torch.nn.functional as F


def precision_at_k(y_true, y_scores, k):
    """
    计算Precision@k
    :param y_true: 真实标签 (n_samples,)
    :param y_scores: 预测概率分数 (n_samples,)
    :param k: 前k个预测
    :return: Precision@k
    """
    # 获取前k个最高分数的索引
    top_k_indices = np.argsort(y_scores)[::-1][:k]
    # 计算前k个中真实正样本的数量
    relevant_in_top_k = np.sum(y_true[top_k_indices])
    return relevant_in_top_k / k


def recall_at_k(y_true, y_scores, k):
    """
    计算Recall@k
    :param y_true: 真实标签 (n_samples,)
    :param y_scores: 预测概率分数 (n_samples,)
    :param k: 前k个预测
    :return: Recall@k
    """
    # 获取前k个最高分数的索引
    top_k_indices = np.argsort(y_scores)[::-1][:k]
    # 计算前k个中真实正样本的数量
    relevant_in_top_k = np.sum(y_true[top_k_indices])
    # 计算总的真实正样本数量
    total_relevant = np.sum(y_true)
    return relevant_in_top_k / total_relevant if total_relevant > 0 else 0.0


def f1_at_k(y_true, y_scores, k):
    """
    计算F1@k
    :param y_true: 真实标签 (n_samples,)
    :param y_scores: 预测概率分数 (n_samples,)
    :param k: 前k个预测
    :return: F1@k
    """
    p_at_k = precision_at_k(y_true, y_scores, k)
    r_at_k = recall_at_k(y_true, y_scores, k)
    if p_at_k + r_at_k == 0:
        return 0.0
    return 2 * p_at_k * r_at_k / (p_at_k + r_at_k)


def ndcg_at_k(y_true, y_scores, k):
    """
    计算NDCG@k
    :param y_true: 真实标签 (n_samples,)
    :param y_scores: 预测概率分数 (n_samples,)
    :param k: 前k个预测
    :return: NDCG@k
    """
    # 将y_true和y_scores转换为2D数组格式，符合sklearn的ndcg_score要求
    y_true_2d = y_true.reshape(1, -1)
    y_scores_2d = y_scores.reshape(1, -1)
    
    try:
        return ndcg_score(y_true_2d, y_scores_2d, k=k)
    except:
        # 如果sklearn的ndcg_score失败，使用自定义实现
        return custom_ndcg_at_k(y_true, y_scores, k)


def custom_ndcg_at_k(y_true, y_scores, k):
    """
    自定义NDCG@k实现
    """
    # 获取前k个最高分数的索引
    top_k_indices = np.argsort(y_scores)[::-1][:k]
    
    # 计算DCG
    dcg = 0.0
    for i, idx in enumerate(top_k_indices):
        dcg += y_true[idx] / np.log2(i + 2)  # log2(i+2) 因为i从0开始
    
    # 计算IDCG (理想情况下的DCG)
    ideal_scores = np.sort(y_true)[::-1][:k]
    idcg = 0.0
    for i, score in enumerate(ideal_scores):
        idcg += score / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0


def test_advanced_metrics(features_by_cluster=None, adjs_by_cluster=None, labels_by_cluster=None, 
                         cluster_data=None, model_path="./model/best_model_multi_cluster.pth", k_values=[5, 10, 20]):
    """
    使用高级指标测试多类别司机模型
    
    :param features_by_cluster: 每个类别的特征字典（可选，如果提供cluster_data则忽略）
    :param adjs_by_cluster: 每个类别的邻接矩阵字典（可选，如果提供cluster_data则忽略）
    :param labels_by_cluster: 每个类别的标签字典（可选，如果提供cluster_data则忽略）
    :param cluster_data: 训练时返回的cluster_data，包含预处理好的数据和划分信息
    :param model_path: 模型路径
    :param k_values: 要测试的k值列表
    """
    hidden_dim = 16
    dropout = 0.2
    alpha = 0.2
    nheads = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 如果提供了cluster_data，直接使用；否则重新构建数据
    if cluster_data is not None:
        print("Using pre-built cluster data from training...")
        test_cluster_data = cluster_data
        # 获取特征维度
        if cluster_data:
            first_cluster_id = next(iter(cluster_data.keys()))
            total_features_dim = cluster_data[first_cluster_id]['data'].x.shape[1]
            print(f"Feature dimension: {total_features_dim}")
        else:
            print("Empty cluster_data provided")
            return None, None
    else:
        print("Building cluster data from scratch...")
        if features_by_cluster is None or adjs_by_cluster is None or labels_by_cluster is None:
            print("Error: features_by_cluster, adjs_by_cluster, and labels_by_cluster must be provided when cluster_data is None")
            return None, None
        # 原有的数据构建逻辑
        test_cluster_data = {}
        total_features_dim = None
        
        for cluster_id in features_by_cluster.keys():
            if cluster_id == 13:  # 跳过类别13
                continue
                
            features = features_by_cluster[cluster_id]
            adjs = adjs_by_cluster[cluster_id]
            labels = labels_by_cluster[cluster_id]
            
            N = len(features)
            
            # 检查特征维度一致性
            if total_features_dim is None:
                total_features_dim = len(features[0]) if features else 0
            else:
                current_dim = len(features[0]) if features else 0
                if current_dim != total_features_dim:
                    # 对每个节点的特征进行填充或截断
                    for i in range(len(features)):
                        if len(features[i]) < total_features_dim:
                            features[i].extend([0] * (total_features_dim - len(features[i])))
                        elif len(features[i]) > total_features_dim:
                            features[i] = features[i][:total_features_dim]
            
            # 将字典格式的邻接表转换为稀疏矩阵
            if isinstance(adjs, dict):
                adj_matrix = np.zeros((N, N), dtype=np.float32)
                for u, neighbors in adjs.items():
                    for v, weight in neighbors.items():
                        if u < N and v < N:
                            adj_matrix[u, v] = weight
                adj = csr_matrix(adj_matrix)
            else:
                adj = csr_matrix(adjs)
                
            features_tensor = torch.tensor(features, dtype=torch.float)
            labels_tensor = torch.tensor(labels, dtype=torch.float)
            
            # 划分训练/验证/测试集
            labels_array = np.array(labels)
            train_idx, test_idx = train_test_split(np.arange(N), test_size=0.2, stratify=labels_array)
            train_idx, val_idx = train_test_split(train_idx, test_size=0.25, stratify=labels_array[train_idx])
            
            test_mask = torch.zeros(N, dtype=torch.bool)
            test_mask[test_idx] = True
            
            edge_index, edge_weight = from_scipy_sparse_matrix(adj)
            
            data = Data(x=features_tensor, edge_index=edge_index, y=labels_tensor, test_mask=test_mask).to(device)
            
            test_cluster_data[cluster_id] = {
                'data': data,
                'test_idx': test_idx
            }
    
    # 打印每个cluster的测试样本信息
    for cluster_id, cluster_info in test_cluster_data.items():
        data = cluster_info['data']
        test_samples = data.test_mask.sum().item()
        positive_samples = data.y[data.test_mask].sum().item()
        print(f'Cluster {cluster_id}: {data.x.shape[0]} nodes, test samples: {test_samples}, positive: {positive_samples}')

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
        return None, None
    
    model.eval()

    print("\n=== Advanced Metrics Testing Results ===")
    
    # 存储所有类别的结果
    all_results = {}
    
    with torch.no_grad():
        for cluster_id, cluster_info in test_cluster_data.items():
            data = cluster_info['data']
            out = model(data.x, data.edge_index)
            scores = torch.sigmoid(out).cpu().numpy()
            
            # 获取测试集的真实标签和预测分数
            test_mask = data.test_mask.cpu().numpy()
            y_true = data.y.cpu().numpy()[test_mask]
            y_scores = scores[test_mask]
            
            print(f"\n--- Cluster {cluster_id} ---")
            print(f"Test samples: {len(y_true)}, Positive samples: {np.sum(y_true)}")
            
            # 计算AUC-ROC
            try:
                auc_roc = roc_auc_score(y_true, y_scores)
                print(f"AUC-ROC: {auc_roc:.4f}")
            except:
                print("AUC-ROC: N/A (insufficient positive samples)")
                auc_roc = 0.0
            
            cluster_results = {'auc_roc': auc_roc}
            
            # 计算不同k值的指标
            for k in k_values:
                if k <= len(y_true):
                    p_at_k = precision_at_k(y_true, y_scores, k)
                    r_at_k = recall_at_k(y_true, y_scores, k)
                    f1_at_k_val = f1_at_k(y_true, y_scores, k)
                    ndcg_at_k_val = ndcg_at_k(y_true, y_scores, k)
                    
                    print(f"k={k}: P@{k}={p_at_k:.4f}, R@{k}={r_at_k:.4f}, F1@{k}={f1_at_k_val:.4f}, NDCG@{k}={ndcg_at_k_val:.4f}")
                    
                    cluster_results[f'p_at_{k}'] = p_at_k
                    cluster_results[f'r_at_{k}'] = r_at_k
                    cluster_results[f'f1_at_{k}'] = f1_at_k_val
                    cluster_results[f'ndcg_at_{k}'] = ndcg_at_k_val
                else:
                    print(f"k={k}: N/A (k > number of test samples)")
                    cluster_results[f'p_at_{k}'] = 0.0
                    cluster_results[f'r_at_{k}'] = 0.0
                    cluster_results[f'f1_at_{k}'] = 0.0
                    cluster_results[f'ndcg_at_{k}'] = 0.0
            
            all_results[cluster_id] = cluster_results
    
    # 计算平均指标
    print("\n=== Average Metrics Across All Clusters ===")
    avg_metrics = {}
    
    # 计算AUC-ROC平均值
    auc_rocs = [results['auc_roc'] for results in all_results.values()]
    avg_auc_roc = np.mean(auc_rocs)
    print(f"Average AUC-ROC: {avg_auc_roc:.4f}")
    avg_metrics['avg_auc_roc'] = avg_auc_roc
    
    # 计算不同k值的平均指标
    for k in k_values:
        p_at_k_vals = [results[f'p_at_{k}'] for results in all_results.values()]
        r_at_k_vals = [results[f'r_at_{k}'] for results in all_results.values()]
        f1_at_k_vals = [results[f'f1_at_{k}'] for results in all_results.values()]
        ndcg_at_k_vals = [results[f'ndcg_at_{k}'] for results in all_results.values()]
        
        avg_p_at_k = np.mean(p_at_k_vals)
        avg_r_at_k = np.mean(r_at_k_vals)
        avg_f1_at_k = np.mean(f1_at_k_vals)
        avg_ndcg_at_k = np.mean(ndcg_at_k_vals)
        
        print(f"Average k={k}: P@{k}={avg_p_at_k:.4f}, R@{k}={avg_r_at_k:.4f}, "
              f"F1@{k}={avg_f1_at_k:.4f}, NDCG@{k}={avg_ndcg_at_k:.4f}")
        
        avg_metrics[f'avg_p_at_{k}'] = avg_p_at_k
        avg_metrics[f'avg_r_at_{k}'] = avg_r_at_k
        avg_metrics[f'avg_f1_at_{k}'] = avg_f1_at_k
        avg_metrics[f'avg_ndcg_at_{k}'] = avg_ndcg_at_k
    
    return all_results, avg_metrics


if __name__ == "__main__":
    # 仅测试流程，加载已训练的模型和数据划分
    from data_processing import data_prepare
    from train import test_multi_cluster
    
    print("=== Starting Testing Pipeline (No Training) ===")
    
    # 数据路径配置
    poi_path = 'E:/Project/traffic/order_data/examine_neighbor_poi_anomaly_detect_chengdu_in_500_1101.txt'
    driver_order_path = 'E:/Project/traffic/order_data/order_driver_01.txt'
    ground_truth_path = 'E:/Project/traffic/order_data/ground_truth.xlsx'
    grid_granularity = 500
    
    # 数据准备
    print("=== Data Preparation ===")
    features_by_cluster, adjs_by_cluster, labels_by_cluster, driver_clusters, cluster_boundaries = data_prepare(
        poi_path, driver_order_path, ground_truth_path, grid_granularity
    )
    
    # 重建训练时的数据结构（使用相同的数据划分逻辑）
    print("=== Rebuilding Training Data Structure ===")
    from sklearn.model_selection import train_test_split
    from torch_geometric.data import Data
    from torch_geometric.utils import from_scipy_sparse_matrix
    from scipy.sparse import csr_matrix
    import torch
    import numpy as np
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cluster_data = {}
    total_features_dim = 782  # 根据训练时的特征维度
    
    for cluster_id in features_by_cluster.keys():
        if cluster_id == 11:
            continue
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
            adj_matrix = np.zeros((N, N), dtype=np.float32)
            for u, neighbors in adjs.items():
                for v, weight in neighbors.items():
                    if u < N and v < N:
                        adj_matrix[u, v] = weight
            adj = csr_matrix(adj_matrix)
        else:
            adj = csr_matrix(adjs)
            
        features_tensor = torch.tensor(features, dtype=torch.float)
        labels_tensor = torch.tensor(labels, dtype=torch.float)
        
        # 使用与训练时相同的数据划分（固定随机种子确保一致性）
        np.random.seed(42)  # 确保数据划分一致
        labels_array = np.array(labels)
        train_idx, test_idx = train_test_split(np.arange(N), test_size=0.2, stratify=labels_array, random_state=42)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.25, stratify=labels_array[train_idx], random_state=42)
        
        train_mask = torch.zeros(N, dtype=torch.bool)
        val_mask = torch.zeros(N, dtype=torch.bool)
        test_mask = torch.zeros(N, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        edge_index, edge_weight = from_scipy_sparse_matrix(adj)
        
        data = Data(x=features_tensor, edge_index=edge_index, y=labels_tensor,
                   train_mask=train_mask, val_mask=val_mask, test_mask=test_mask).to(device)
        
        cluster_data[cluster_id] = {
            'data': data,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        }
        
        print(f'Cluster {cluster_id}: {N} nodes, test samples: {len(test_idx)}, positive: {labels_tensor[test_idx].sum().item()}')
    
    # 测试多类别模型（传统指标）
    print("\n=== Multi-Cluster Testing (Traditional Metrics) ===")
    test_multi_cluster(features_by_cluster, adjs_by_cluster, labels_by_cluster)
    
    # 测试多类别模型（高级指标）- 使用重建的数据结构
    print("\n=== Multi-Cluster Testing (Advanced Metrics) ===")
    try:
        results, avg_metrics = test_advanced_metrics(cluster_data=cluster_data)
        
        if results is not None and avg_metrics is not None:
            # 输出最终结果摘要
            print("\n=== Final Results Summary ===")
            print(f"Average AUC-ROC: {avg_metrics['avg_auc_roc']:.4f}")
            for k in [5, 10, 20]:
                print(f"Average F1@{k}: {avg_metrics[f'avg_f1_at_{k}']:.4f}")
                print(f"Average NDCG@{k}: {avg_metrics[f'avg_ndcg_at_{k}']:.4f}")
            
            # 检查是否达到目标F1分数
            target_f1 = 0.6
            best_f1_at_k = max([avg_metrics[f'avg_f1_at_{k}'] for k in [5, 10, 20]])
            
            print(f"\n=== Performance Analysis ===")
            print(f"Best F1@k score: {best_f1_at_k:.4f}")
            print(f"Target F1 score: {target_f1:.4f}")
            
            if best_f1_at_k >= target_f1:
                print("🎉 SUCCESS: Target F1 score achieved!")
            else:
                print(f"❌ Target not reached. Need improvement of {target_f1 - best_f1_at_k:.4f}")
                print("\nSuggested optimizations for train.py:")
                print("1. Increase learning rate from 5e-4 to 1e-3")
                print("2. Increase hidden dimensions from 16 to 32 or 64")
                print("3. Increase attention heads from 8 to 12 or 16")
                print("4. Adjust loss function weights (try 0.3 focal + 0.7 f1)")
                print("5. Reduce dropout from 0.2 to 0.1")
                print("6. Increase training epochs if early stopping")
        else:
            print("Advanced metrics testing returned None")
    except Exception as e:
        print(f"Advanced metrics testing failed: {e}")
        import traceback
        traceback.print_exc() 