# 出租车司机异常行为检测系统

基于图神经网络的出租车司机异常驾驶模式检测系统，使用GPS轨迹数据和POI信息进行智能分析。

## 🚀 功能特点

- **图注意力网络（GAT）** 进行异常检测
- **多类别司机建模** 基于舒适区域分析
- **POI特征融合** 增强空间理解能力
- **稀疏图处理** 提高计算效率
- **网格化空间分析** 支持可配置粒度
- **多种损失函数** （Focal Loss、F1 Loss、Precision Loss）
- **数据增强** 和缓存机制

## 📋 系统要求

- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric
- NumPy, Pandas, Scikit-learn
- Matplotlib, Folium (用于可视化)
- SciPy

## 🛠️ 安装指南

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/taxi-anomaly-detection.git
cd taxi-anomaly-detection
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 设置数据目录：
```bash
mkdir -p data/model data/similarity_matrix data/augmented_data_v2
```

## 📊 数据格式

### 输入数据要求

1. **POI数据** (`poi_path`):
   - 格式：包含POI信息的文本文件
   - 列：站点坐标、POI类型和数量
   - 示例：`examine_neighbor_poi_anomaly_detect_chengdu_in_500_1101.txt`

2. **司机订单数据** (`driver_order_path`):
   - 格式：包含司机轨迹的CSV文件
   - 列：司机ID、订单ID、上下车坐标和时间戳
   - 示例：`order_driver_01.txt`

3. **真实标签** (`ground_truth_path`):
   - 格式：包含标记异常位置的Excel文件
   - 列：坐标和标签
   - 示例：`ground_truth.xlsx`

### 数据结构
```
data/
├── model/                    # 训练好的模型
├── similarity_matrix/        # 司机相似度矩阵
├── augmented_data_v2/        # 增强的司机数据
└── cluster_data_cache_v2.pkl # 缓存的聚类数据
```

## 🚀 快速开始

### 基本使用

```python
from data_processing import data_prepare
from train import train

# 准备数据
poi_path = 'path/to/poi_data.txt'
driver_order_path = 'path/to/driver_orders.txt'
ground_truth_path = 'path/to/ground_truth.xlsx'
grid_granularity = 500

# 加载和处理数据
features_by_cluster, adjs_by_cluster, labels_by_cluster, driver_clusters, cluster_boundaries = data_prepare(
    poi_path, driver_order_path, ground_truth_path, grid_granularity
)

# 训练模型
model, cluster_data = train(features_by_cluster, adjs_by_cluster, labels_by_cluster)
```

### 高级使用

```python
# 网格压缩以提高效率
from main import compress_grid_data

# 将网格数据压缩到更小区域
compressed_features = compress_grid_data(features_by_cluster)
compressed_labels = compress_grid_data(labels_by_cluster)

# 使用压缩数据训练
model, cluster_data = train(compressed_features, adjs_by_cluster, compressed_labels)
```

## 📈 模型架构

### 稀疏图注意力网络（SparseGAT）

- **输入**：节点特征（POI + 司机轨迹特征）
- **图结构**：空间邻接 + 司机轨迹连接
- **注意力机制**：多头注意力进行邻居聚合
- **输出**：二分类（异常/正常）

### 核心组件

1. **SparseGraphAttentionLayer**：核心注意力机制
2. **多头注意力**：8个注意力头进行鲁棒特征学习
3. **Dropout正则化**：0.2的dropout率
4. **LeakyReLU激活**：α=0.2的负斜率

## 🔧 配置说明

### 模型参数

```python
# 模型超参数
hidden_dim = 16          # 隐藏层维度
dropout = 0.2           # Dropout率
alpha = 0.2             # LeakyReLU负斜率
nheads = 8              # 注意力头数量
lr = 5e-4              # 学习率
weight_decay = 5e-4     # 权重衰减
epochs = 10000         # 训练轮数
```

### 网格配置

```python
# 网格边界（根据你的数据调整）
poi_boundary = [-28, -75, 73, 29]  # [min_x, min_y, max_x, max_y]
grid_granularity = 500             # 网格大小（米）
```

## 📊 性能指标

系统使用以下指标评估性能：

- **准确率**：整体分类准确率
- **F1分数**：精确率和召回率的调和平均数
- **精确率**：真正例 / (真正例 + 假正例)
- **召回率**：真正例 / (真正例 + 假负例)

### 示例结果
```
阈值: 0.5000, 平均测试准确率: 0.8542, 
平均F1: 0.7234, 平均精确率: 0.6891, 平均召回率: 0.7612
```

## 🗂️ 项目结构

```
├── data_processing.py      # 数据加载和预处理
├── model.py               # SparseGAT模型定义
├── train.py                # 训练和评估
├── main.py                 # 主执行脚本
├── gps_analysis.py         # GPS轨迹分析
├── utils.py                # 工具函数
├── preprocess.py           # 数据预处理工具
├── test_advanced.py        # 高级测试
├── example_multi_cluster.py # 多聚类示例
├── config.py               # 配置文件
├── examples/               # 使用示例
│   ├── basic_usage.py      # 基本使用示例
│   ├── advanced_usage.py   # 高级使用示例
│   └── visualization_example.py # 可视化示例
├── data/                   # 数据目录
│   ├── model/             # 模型文件
│   ├── similarity_matrix/ # 相似度矩阵
│   └── augmented_data_v2/ # 增强数据
└── README_CN.md           # 中文说明文档
```

## 🔬 研究应用

本系统可用于：

- **交通管理**：监控出租车司机行为模式
- **异常检测**：识别异常驾驶模式
- **城市规划**：理解城市交通流
- **政策制定**：数据驱动的交通政策


## 🐛 故障排除

### 常见问题

1. **CUDA内存不足**：减少批次大小或使用CPU
2. **数据格式错误**：检查输入数据格式和编码
3. **缺少依赖**：从requirements.txt安装所有依赖

### 性能优化建议

- 在可用时使用GPU加速
- 为重复运行启用数据缓存
- 对大规模分析使用网格数据压缩

## 🔄 更新日志

### 版本 1.0.0
- 初始发布
- SparseGAT实现
- 多类别司机建模
- POI特征融合
- 基于网格的空间分析

---

**注意**：本系统设计用于研究目的。使用真实世界数据时，请确保符合当地数据隐私法规。
