"""
多类别司机休息区域推荐模型示例

这个脚本演示了如何使用修改后的train函数来训练多类别司机模型。
模型会同时学习所有司机类别的偏好，通过最小化总损失来优化推荐策略。
"""

from data_processing import data_prepare
from train import train, test_multi_cluster
import torch


def main():
    # 数据路径配置
    poi_path = 'D:/Project/traffic/order_data/examine_neighbor_poi_anomaly_detect_chengdu_in_500_1101.txt'
    driver_order_path = 'D:/Project/traffic/order_data/order_driver_01.txt'
    ground_truth_path = 'D:/Project/traffic/order_data/ground_truth.xlsx'
    grid_granularity = 500

    print("=== 多类别司机休息区域推荐模型 ===")
    print("1. 数据准备阶段...")
    
    # 准备多类别数据
    features_by_cluster, adjs_by_cluster, labels_by_cluster, driver_clusters, cluster_boundaries = data_prepare(
        poi_path, driver_order_path, ground_truth_path, grid_granularity
    )
    
    print(f"数据准备完成！共有 {len(features_by_cluster)} 个司机类别")
    for cluster_id in features_by_cluster.keys():
        print(f"  类别 {cluster_id}: {len(features_by_cluster[cluster_id])} 个节点")
    
    print("\n2. 模型训练阶段...")
    print("开始多类别联合训练，模型将同时学习所有司机类别的偏好...")
    
    # 训练多类别模型
    model, cluster_data = train(
        features_by_cluster=features_by_cluster,
        adjs_by_cluster=adjs_by_cluster,
        labels_by_cluster=labels_by_cluster
    )
    
    print("\n3. 模型测试阶段...")
    print("测试模型在所有司机类别上的表现...")
    
    # 测试多类别模型
    test_multi_cluster(features_by_cluster, adjs_by_cluster, labels_by_cluster)
    
    print("\n=== 训练完成 ===")
    print("模型已保存到 ./model/best_model_multi_cluster.pth")
    print("该模型能够为不同类别的司机提供个性化的休息区域推荐")


if __name__ == '__main__':
    main() 