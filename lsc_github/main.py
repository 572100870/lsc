from data_processing import data_prepare
from train import train, test_multi_cluster
import torch
import numpy as np


def compress_grid_data(data, original_x_range=(-89, 146), original_y_range=(-107, 68), 
                      new_x_range=(-28, 73), new_y_range=(-75, 29)):
    # 计算原始和新的网格尺寸
    orig_x_size = original_x_range[1] - original_x_range[0] + 1  # 236
    orig_y_size = original_y_range[1] - original_y_range[0] + 1  # 176
    new_x_size = new_x_range[1] - new_x_range[0] + 1  # 102
    new_y_size = new_y_range[1] - new_y_range[0] + 1  # 105
    
    print(f"原始网格尺寸: {orig_x_size} x {orig_y_size} = {orig_x_size * orig_y_size}")
    print(f"新网格尺寸: {new_x_size} x {new_y_size} = {new_x_size * new_y_size}")
    
    # 计算新范围在原始范围中的起始索引
    x_start_idx = new_x_range[0] - original_x_range[0]  # -28 - (-89) = 61
    y_start_idx = new_y_range[0] - original_y_range[0]  # -75 - (-107) = 32
    
    print(f"提取区域起始索引: x={x_start_idx}, y={y_start_idx}")
    
    if isinstance(data, dict):
        # 处理按簇分组的数据
        compressed_data = {}
        for cluster_id, cluster_data in data.items():
            if cluster_id == 13:
                continue
            compressed_data[cluster_id] = compress_single_data(
                cluster_data, orig_x_size, orig_y_size, new_x_size, new_y_size,
                x_start_idx, y_start_idx
            )
        return compressed_data
    else:
        # 处理单个数据
        return compress_single_data(
            data, orig_x_size, orig_y_size, new_x_size, new_y_size,
            x_start_idx, y_start_idx
        )


def compress_single_data(data, orig_x_size, orig_y_size, new_x_size, new_y_size,
                        x_start_idx, y_start_idx):
    """压缩单个数据项"""
    if not isinstance(data[0], int):
        total_features_dim = 782
        for i in range(len(data)):
            if len(data[i]) < total_features_dim:
                data[i].extend([0] * (total_features_dim - len(data[i])))
            elif len(data[i]) > total_features_dim:
                data[i] = data[i][:total_features_dim]
        
    if isinstance(data, torch.Tensor):
        is_tensor = True
        data_np = data.cpu().numpy()
    else:
        is_tensor = False
        data_np = np.array(data)
    
    if data_np.ndim == 1:
        # 一维向量：重塑为2D网格，提取子区域，再展平
        if len(data_np) != orig_x_size * orig_y_size:
            raise ValueError(f"数据长度 {len(data_np)} 与预期的网格大小 {orig_x_size * orig_y_size} 不匹配")
        
        # 重塑为2D网格 (y_size, x_size) - 注意行列对应关系
        grid_2d = data_np.reshape(orig_y_size, orig_x_size)
        
        # 提取子区域
        sub_grid = grid_2d[y_start_idx:y_start_idx + new_y_size, 
                          x_start_idx:x_start_idx + new_x_size]
        
        # 展平为一维向量
        compressed = sub_grid.flatten()
        
    elif data_np.ndim == 2:
        # 判断是特征矩阵还是邻接矩阵
        if data_np.shape[0] == orig_x_size * orig_y_size and data_np.shape[1] != orig_x_size * orig_y_size:
            # 特征矩阵：形状为 (节点数, 特征维度)，只压缩第一个维度
            print(f"处理特征矩阵，原始形状: {data_np.shape}")
            
            # 创建新的索引映射
            old_indices = []
            for y in range(y_start_idx, y_start_idx + new_y_size):
                for x in range(x_start_idx, x_start_idx + new_x_size):
                    old_idx = y * orig_x_size + x
                    old_indices.append(old_idx)
            
            # 只提取对应的行（节点），保持特征维度不变
            compressed = data_np[old_indices, :]
            print(f"压缩后特征矩阵形状: {compressed.shape}")
            
        elif data_np.shape[0] == orig_x_size * orig_y_size and data_np.shape[1] == orig_x_size * orig_y_size:
            # 邻接矩阵：需要同时压缩行和列
            print(f"处理邻接矩阵，原始形状: {data_np.shape}")
            
            # 创建新旧索引的映射
            old_indices = []
            for y in range(y_start_idx, y_start_idx + new_y_size):
                for x in range(x_start_idx, x_start_idx + new_x_size):
                    old_idx = y * orig_x_size + x
                    old_indices.append(old_idx)
            
            # 提取对应的行和列
            compressed = data_np[np.ix_(old_indices, old_indices)]
            print(f"压缩后邻接矩阵形状: {compressed.shape}")
            
        else:
            raise ValueError(f"二维矩阵形状 {data_np.shape} 不符合预期格式")
        
    else:
        raise ValueError(f"不支持的数据维度: {data_np.ndim}")
    
    # 转换回原始数据类型
    if is_tensor:
        return torch.from_numpy(compressed).to(data.device)
    else:
        return compressed


if __name__ == '__main__':
    poi_path = 'E:/Project/traffic/order_data/examine_neighbor_poi_anomaly_detect_chengdu_in_500_1101.txt'
    driver_order_path = 'E:/Project/traffic/order_data/order_driver_01.txt'
    ground_truth_path = 'E:/Project/traffic/order_data/ground_truth.xlsx'
    grid_granularity = 500

    # features, adjs, label = data_prepare(poi_path, driver_order_path, ground_truth_path, grid_granularity)
    # train(features, adjs, label)
    features_by_cluster, adjs_by_cluster, labels_by_cluster, driver_clusters, cluster_boundaries = data_prepare(poi_path, driver_order_path, \
                                                           ground_truth_path, grid_granularity)
    
    # 压缩网格数据到新范围
    # print("=== 开始压缩网格数据 ===")
    # compressed_features = compress_grid_data(features_by_cluster)
    # # compressed_adjs = compress_grid_data(adjs_by_cluster) 
    # compressed_labels = compress_grid_data(labels_by_cluster)

    # print("压缩完成！")
    # print(f"压缩后各簇的特征维度:")
    # if isinstance(compressed_features, dict):
    #     for cluster_id, features in compressed_features.items():
    #         print(f"  簇 {cluster_id}: {features.shape}")
    # else:
    #     print(f"  压缩后特征维度: {compressed_features.shape}")
    
    # 训练多类别模型
    print("\n=== Starting Multi-Cluster Training ===")
    model, cluster_data = train(features_by_cluster, adjs_by_cluster, labels_by_cluster)
    # 测试多类别模型
    # print("\n=== Starting Multi-Cluster Testing ===")
    # test_multi_cluster(features_by_cluster, adjs_by_cluster, labels_by_cluster)
    # train()
