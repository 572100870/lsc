"""
出租车司机异常行为检测系统 - 可视化示例
演示如何可视化聚类区域、司机分布和性能指标
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
from gps_analysis import get_coordinate, get_distance
from config import get_config

def visualize_cluster_regions(driver_clusters, cluster_boundaries, driver_id_list):
    """
    可视化聚类区域和司机分布
    
    参数:
        driver_clusters (dict): 司机聚类结果
        cluster_boundaries (dict): 聚类边界
        driver_id_list (list): 司机ID列表
    """
    
    print("=== 聚类可视化 ===")
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 图1: 聚类边界
    colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_boundaries)))
    
    for i, (cluster_id, boundary) in enumerate(cluster_boundaries.items()):
        if boundary and len(boundary) >= 3:
            # 绘制凸包
            hull_x = [point[0] for point in boundary] + [boundary[0][0]]
            hull_y = [point[1] for point in boundary] + [boundary[0][1]]
            ax1.plot(hull_x, hull_y, color=colors[i], linewidth=2, 
                    label=f'聚类 {cluster_id}')
            ax1.fill(hull_x, hull_y, color=colors[i], alpha=0.3)
    
    ax1.set_title('聚类区域')
    ax1.set_xlabel('X坐标')
    ax1.set_ylabel('Y坐标')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2: 司机分布
    cluster_sizes = [len(driver_clusters[cluster_id]) for cluster_id in driver_clusters.keys()]
    cluster_ids = list(driver_clusters.keys())
    
    bars = ax2.bar(cluster_ids, cluster_sizes, color=colors[:len(cluster_ids)])
    ax2.set_title('各聚类的司机分布')
    ax2.set_xlabel('聚类ID')
    ax2.set_ylabel('司机数量')
    ax2.set_xticks(cluster_ids)
    
    # 在柱状图上添加数值标签
    for bar, size in zip(bars, cluster_sizes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(size), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('cluster_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ 聚类可视化已保存为 'cluster_visualization.png'")

def visualize_trajectory_patterns(driver_data, sample_size=5):
    """
    可视化样本司机轨迹模式
    
    参数:
        driver_data (dict): 司机数据
        sample_size (int): 样本数量
    """
    
    print("=== 轨迹模式可视化 ===")
    
    # 采样司机
    driver_ids = list(driver_data.keys())[:sample_size]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, driver_id in enumerate(driver_ids):
        if i >= len(axes):
            break
            
        ax = axes[i]
        orders = driver_data[driver_id]
        
        # 绘制上下车点
        pickup_x = [order[4] for order in orders]
        pickup_y = [order[5] for order in orders]
        dropoff_x = [order[6] for order in orders]
        dropoff_y = [order[7] for order in orders]
        
        # 绘制上车点
        ax.scatter(pickup_x, pickup_y, c='green', marker='o', s=50, 
                  label='上车点', alpha=0.7)
        
        # 绘制下车点
        ax.scatter(dropoff_x, dropoff_y, c='red', marker='s', s=50, 
                  label='下车点', alpha=0.7)
        
        # 绘制上下车点之间的连线
        for order in orders:
            ax.plot([order[4], order[6]], [order[5], order[7]], 
                   'b-', alpha=0.5, linewidth=1)
        
        ax.set_title(f'司机 {driver_id[:8]}...')
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 隐藏未使用的子图
    for i in range(len(driver_ids), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('trajectory_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ 轨迹模式已保存为 'trajectory_patterns.png'")

def visualize_performance_metrics():
    """
    可视化模型性能指标
    
    展示不同阈值下的模型性能表现
    """
    
    print("=== 性能指标可视化 ===")
    
    # 示例性能数据（替换为实际结果）
    thresholds = [0.5 + 0.05 * i for i in range(10)]
    accuracy = [0.85, 0.87, 0.89, 0.91, 0.90, 0.88, 0.86, 0.84, 0.82, 0.80]
    f1_scores = [0.72, 0.75, 0.78, 0.80, 0.79, 0.77, 0.75, 0.73, 0.71, 0.69]
    precision = [0.68, 0.71, 0.74, 0.76, 0.75, 0.73, 0.71, 0.69, 0.67, 0.65]
    recall = [0.76, 0.79, 0.82, 0.84, 0.83, 0.81, 0.79, 0.77, 0.75, 0.73]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 准确率 vs 阈值
    ax1.plot(thresholds, accuracy, 'b-o', linewidth=2, markersize=6)
    ax1.set_title('准确率 vs 阈值')
    ax1.set_xlabel('阈值')
    ax1.set_ylabel('准确率')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.8, 0.95)
    
    # F1分数 vs 阈值
    ax2.plot(thresholds, f1_scores, 'g-s', linewidth=2, markersize=6)
    ax2.set_title('F1分数 vs 阈值')
    ax2.set_xlabel('阈值')
    ax2.set_ylabel('F1分数')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.65, 0.85)
    
    # 精确率 vs 阈值
    ax3.plot(thresholds, precision, 'r-^', linewidth=2, markersize=6)
    ax3.set_title('精确率 vs 阈值')
    ax3.set_xlabel('阈值')
    ax3.set_ylabel('精确率')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.6, 0.8)
    
    # 召回率 vs 阈值
    ax4.plot(thresholds, recall, 'm-d', linewidth=2, markersize=6)
    ax4.set_title('召回率 vs 阈值')
    ax4.set_xlabel('阈值')
    ax4.set_ylabel('召回率')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0.7, 0.9)
    
    plt.tight_layout()
    plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ 性能指标已保存为 'performance_metrics.png'")

def main():
    """
    主可视化示例
    
    演示各种可视化功能
    """
    
    print("=== 可视化示例 ===")
    
    # 注意：这些示例需要实际数据
    # 在实际使用中，你需要先加载数据
    
    print("\n1. 性能指标可视化")
    visualize_performance_metrics()
    
    print("\n=== 可视化示例完成！ ===")
    print("注意：对于聚类和轨迹可视化，你需要先运行数据处理。")

if __name__ == "__main__":
    main()
