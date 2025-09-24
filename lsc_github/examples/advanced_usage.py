"""
出租车司机异常行为检测系统 - 高级使用示例
演示网格压缩、自定义配置和模型分析等高级功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing import data_prepare
from train import train
from main import compress_grid_data
from config import get_config
import torch

def advanced_example():
    """
    高级使用示例
    
    演示以下高级功能：
    1. 自定义模型配置
    2. 网格数据压缩
    3. 模型性能分析
    """
    
    # 获取配置
    config = get_config('development')
    
    # 自定义配置
    config.EPOCHS = 1000  # 演示用，减少训练轮数
    config.HIDDEN_DIM = 32  # 更大的隐藏层维度
    config.LEARNING_RATE = 1e-3  # 更高的学习率
    
    print("=== 高级出租车司机异常行为检测 ===")
    print(f"自定义训练轮数: {config.EPOCHS}")
    print(f"隐藏层维度: {config.HIDDEN_DIM}")
    print(f"学习率: {config.LEARNING_RATE}")
    
    # 数据路径
    poi_path = config.POI_PATH
    driver_order_path = config.DRIVER_ORDER_PATH
    ground_truth_path = config.GROUND_TRUTH_PATH
    grid_granularity = config.GRID_GRANULARITY
    
    # 步骤1: 加载数据
    print("\n1. 加载数据...")
    features_by_cluster, adjs_by_cluster, labels_by_cluster, driver_clusters, cluster_boundaries = data_prepare(
        poi_path, driver_order_path, ground_truth_path, grid_granularity
    )
    
    # 步骤2: 网格压缩以提高效率
    print("\n2. 压缩网格数据...")
    try:
        compressed_features = compress_grid_data(features_by_cluster)
        compressed_labels = compress_grid_data(labels_by_cluster)
        print("✓ 网格压缩完成")
        
        # 打印压缩统计信息
        for cluster_id in compressed_features.keys():
            if cluster_id in features_by_cluster:
                orig_shape = features_by_cluster[cluster_id].shape if hasattr(features_by_cluster[cluster_id], 'shape') else len(features_by_cluster[cluster_id])
                comp_shape = compressed_features[cluster_id].shape if hasattr(compressed_features[cluster_id], 'shape') else len(compressed_features[cluster_id])
                print(f"  聚类 {cluster_id}: {orig_shape} -> {comp_shape}")
    except Exception as e:
        print(f"✗ 网格压缩失败: {e}")
        # 如果压缩失败，使用原始数据
        compressed_features = features_by_cluster
        compressed_labels = labels_by_cluster
    
    # 步骤3: 使用压缩数据训练
    print("\n3. 使用压缩数据训练...")
    try:
        model, cluster_data = train(compressed_features, adjs_by_cluster, compressed_labels)
        print("✓ 训练完成")
    except Exception as e:
        print(f"✗ 训练失败: {e}")
        return
    
    # 步骤4: 模型分析
    print("\n4. 模型分析...")
    try:
        # 统计参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  总参数数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        
        # 模型设备
        device = next(model.parameters()).device
        print(f"  模型设备: {device}")
        
        # 聚类统计
        for cluster_id, data in cluster_data.items():
            if 'data' in data:
                num_nodes = data['data'].x.shape[0]
                num_edges = data['data'].edge_index.shape[1]
                print(f"  聚类 {cluster_id}: {num_nodes} 个节点, {num_edges} 条边")
    except Exception as e:
        print(f"✗ 模型分析失败: {e}")
    
    print("\n=== 高级示例完成！ ===")

def custom_model_example():
    """
    自定义模型参数示例
    
    演示如何配置自定义的模型参数
    """
    
    print("\n=== 自定义模型配置 ===")
    
    # 自定义模型配置
    custom_config = {
        'hidden_dim': 64,      # 隐藏层维度
        'dropout': 0.3,        # Dropout率
        'alpha': 0.1,          # LeakyReLU负斜率
        'nheads': 16,          # 注意力头数量
        'lr': 1e-4,            # 学习率
        'weight_decay': 1e-3,   # 权重衰减
        'epochs': 500          # 训练轮数
    }
    
    print("自定义配置:")
    for key, value in custom_config.items():
        print(f"  {key}: {value}")
    
    # 你需要修改train函数来接受这些参数
    # 现在只是展示配置
    print("✓ 自定义配置已准备好用于训练")

if __name__ == "__main__":
    advanced_example()
    custom_model_example()
