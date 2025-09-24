# API 文档

本文档详细说明了出租车司机异常行为检测系统的API接口和使用方法。

## 📚 核心模块

### 1. 数据预处理模块 (`data_processing.py`)

#### `poi_processing(poi_path, grid_granularity)`
处理POI数据，提取兴趣点特征。

**参数:**
- `poi_path` (str): POI数据文件路径
- `grid_granularity` (int): 网格粒度（米）

**返回:**
- `poi_data` (dict): POI数据字典
- `poi_boundary` (list): POI边界坐标

**示例:**
```python
poi_data, poi_boundary = poi_processing('data/poi.txt', 500)
```

#### `driver_order_processing(driver_order_path, grid_granularity, poi_boundary)`
处理司机订单数据，进行司机聚类。

**参数:**
- `driver_order_path` (str): 司机订单文件路径
- `grid_granularity` (int): 网格粒度
- `poi_boundary` (list): POI边界

**返回:**
- `driver_data` (dict): 司机数据字典
- `driver_boundary` (list): 司机数据边界
- `driver_clusters` (dict): 司机聚类结果
- `cluster_boundaries` (dict): 聚类边界

#### `build_features(pois, driver_orders, dimension_information)`
构建节点特征矩阵。

**参数:**
- `pois` (dict): POI数据
- `driver_orders` (dict): 司机订单数据
- `dimension_information` (list): 维度信息

**返回:**
- `features` (list): 特征矩阵

#### `build_adjs(driver_orders, dimension_information, sparse)`
构建邻接矩阵。

**参数:**
- `driver_orders` (dict): 司机订单数据
- `dimension_information` (list): 维度信息
- `sparse` (bool): 是否使用稀疏矩阵

**返回:**
- `adjs` (dict): 邻接矩阵字典

### 2. 模型定义模块 (`model.py`)

#### `SparseGraphAttentionLayer`
稀疏图注意力层类。

**初始化参数:**
- `in_features` (int): 输入特征维度
- `out_features` (int): 输出特征维度
- `dropout` (float): Dropout概率
- `alpha` (float): LeakyReLU负斜率
- `concat` (bool): 是否拼接输出

**方法:**
- `forward(x, edge_index)`: 前向传播

#### `SparseGAT`
稀疏图注意力网络类。

**初始化参数:**
- `nfeat` (int): 输入特征维度
- `nhid` (int): 隐藏层维度
- `nclass` (int): 输出类别数
- `dropout` (float): Dropout概率
- `alpha` (float): LeakyReLU负斜率
- `nheads` (int): 注意力头数量

### 3. 训练模块 (`train.py`)

#### `train(features_by_cluster, adjs_by_cluster, labels_by_cluster)`
训练多类别模型。

**参数:**
- `features_by_cluster` (dict): 按聚类分组的特征
- `adjs_by_cluster` (dict): 按聚类分组的邻接矩阵
- `labels_by_cluster` (dict): 按聚类分组的标签

**返回:**
- `model`: 训练好的模型
- `cluster_data`: 聚类数据

#### `test_multi_cluster(features_by_cluster, adjs_by_cluster, labels_by_cluster, model_path)`
测试多类别模型。

**参数:**
- `features_by_cluster` (dict): 按聚类分组的特征
- `adjs_by_cluster` (dict): 按聚类分组的邻接矩阵
- `labels_by_cluster` (dict): 按聚类分组的标签
- `model_path` (str): 模型文件路径

### 4. 主程序模块 (`main.py`)

#### `compress_grid_data(data, original_x_range, original_y_range, new_x_range, new_y_range)`
压缩网格数据到更小区域。

**参数:**
- `data`: 原始数据
- `original_x_range` (tuple): 原始X范围
- `original_y_range` (tuple): 原始Y范围
- `new_x_range` (tuple): 新X范围
- `new_y_range` (tuple): 新Y范围

**返回:**
- 压缩后的数据

### 5. 工具模块 (`utils.py`)

#### `get_coordinate(lon, lat, grid_granularity)`
将经纬度坐标转换为网格坐标。

**参数:**
- `lon` (float): 经度
- `lat` (float): 纬度
- `grid_granularity` (int): 网格粒度

**返回:**
- `(x, y)`: 网格坐标

#### `graham_scan(points)`
计算点集的凸包。

**参数:**
- `points` (list): 点坐标列表

**返回:**
- `convex_hull` (list): 凸包顶点

#### `convex_hull_iou(hull1, hull2)`
计算两个凸包的IoU。

**参数:**
- `hull1` (list): 第一个凸包
- `hull2` (list): 第二个凸包

**返回:**
- `iou` (float): IoU值

## 🔧 配置模块 (`config.py`)

### `Config` 类
主配置类，包含所有系统配置参数。

**主要属性:**
- `BASE_PATH`: 基础路径
- `DATA_PATH`: 数据路径
- `MODEL_PATH`: 模型路径
- `GRID_GRANULARITY`: 网格粒度
- `HIDDEN_DIM`: 隐藏层维度
- `DROPOUT`: Dropout概率
- `LEARNING_RATE`: 学习率
- `EPOCHS`: 训练轮数

### `ModelConfig` 类
模型特定配置。

### `DataConfig` 类
数据处理配置。

## 📊 使用示例

### 基本使用流程

```python
from data_processing import data_prepare
from train import train, test_multi_cluster
from config import get_config

# 获取配置
config = get_config('development')

# 准备数据
features_by_cluster, adjs_by_cluster, labels_by_cluster, driver_clusters, cluster_boundaries = data_prepare(
    config.POI_PATH, 
    config.DRIVER_ORDER_PATH, 
    config.GROUND_TRUTH_PATH, 
    config.GRID_GRANULARITY
)

# 训练模型
model, cluster_data = train(features_by_cluster, adjs_by_cluster, labels_by_cluster)

# 测试模型
test_multi_cluster(features_by_cluster, adjs_by_cluster, labels_by_cluster)
```

### 高级使用

```python
from main import compress_grid_data

# 压缩网格数据
compressed_features = compress_grid_data(features_by_cluster)
compressed_labels = compress_grid_data(labels_by_cluster)

# 使用压缩数据训练
model, cluster_data = train(compressed_features, adjs_by_cluster, compressed_labels)
```

## 🎯 损失函数

### `FocalLoss`
Focal Loss用于处理类别不平衡问题。

**参数:**
- `alpha` (float): 权重参数
- `gamma` (float): 聚焦参数
- `reduction` (str): 归约方式

### `SoftF1Loss`
软F1损失函数。

### `SoftPrecisionLoss`
软精确率损失函数。

## 📈 性能指标

系统支持以下性能指标：

- **准确率** (Accuracy)
- **精确率** (Precision)
- **召回率** (Recall)
- **F1分数** (F1-Score)

## 🔍 数据缓存

系统支持数据缓存以提高效率：

- `save_cache(data, filepath)`: 保存缓存
- `load_cache(filepath)`: 加载缓存
- `get_cache_filepath(base_path, ...)`: 获取缓存文件路径

## ⚠️ 注意事项

1. **内存使用**: 大规模数据可能需要大量内存
2. **GPU支持**: 建议使用GPU加速训练
3. **数据格式**: 确保输入数据格式正确
4. **路径配置**: 根据实际环境调整文件路径

## 🐛 错误处理

常见错误及解决方案：

1. **CUDA内存不足**: 减少批次大小或使用CPU
2. **数据格式错误**: 检查输入数据格式
3. **文件路径错误**: 确认文件路径正确
4. **依赖缺失**: 安装所有必需的依赖包

## 📞 技术支持

如有问题，请：

1. 查看本文档
2. 检查示例代码
3. 提交GitHub Issue
4. 联系开发团队
