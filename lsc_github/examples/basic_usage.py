"""
出租车司机异常行为检测系统 - 基本使用示例
演示如何使用系统进行数据预处理、模型训练和测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing import data_prepare
from train import train, test_multi_cluster
from config import get_config

def main():
    """
    基本使用示例
    
    演示完整的异常检测流程：
    1. 数据加载和预处理
    2. 模型训练
    3. 模型测试
    """
    
    # 获取配置
    config = get_config('development')
    config.create_directories()
    
    # 数据路径（根据你的环境更新这些路径）
    poi_path = config.POI_PATH
    driver_order_path = config.DRIVER_ORDER_PATH
    ground_truth_path = config.GROUND_TRUTH_PATH
    grid_granularity = config.GRID_GRANULARITY
    
    print("=== 出租车司机异常行为检测系统 ===")
    print(f"网格粒度: {grid_granularity}米")
    print(f"POI边界: {config.POI_BOUNDARY}")
    
    # 步骤1: 准备数据
    print("\n1. 加载和处理数据...")
    try:
        features_by_cluster, adjs_by_cluster, labels_by_cluster, driver_clusters, cluster_boundaries = data_prepare(
            poi_path, driver_order_path, ground_truth_path, grid_granularity
        )
        print(f"✓ 数据加载成功")
        print(f"  - 聚类数量: {len(features_by_cluster)}")
        print(f"  - 聚类ID: {list(features_by_cluster.keys())}")
    except Exception as e:
        print(f"✗ 数据加载错误: {e}")
        return
    
    # 步骤2: 训练模型
    print("\n2. 训练多聚类模型...")
    try:
        model, cluster_data = train(features_by_cluster, adjs_by_cluster, labels_by_cluster)
        print("✓ 模型训练完成")
    except Exception as e:
        print(f"✗ 训练过程中出错: {e}")
        return
    
    # 步骤3: 测试模型
    print("\n3. 测试模型...")
    try:
        test_multi_cluster(features_by_cluster, adjs_by_cluster, labels_by_cluster)
        print("✓ 模型测试完成")
    except Exception as e:
        print(f"✗ 测试过程中出错: {e}")
        return
    
    print("\n=== 处理完成！ ===")

if __name__ == "__main__":
    main()
