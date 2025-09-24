import math
import os
import pickle
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import hsv_to_rgb


def get_distance(lat1, lon1, lat2, lon2):
    # 给定两个点的经纬度得到两个点之间的距离
    ra = 6378140  # 赤道半径，单位m
    rb = 6356755  # 极半径，单位m
    flatten = (ra - rb) / ra
    # 角度变成弧度
    rad_lat1 = math.radians(lat1)
    rad_lon1 = math.radians(lon1)
    rad_lat2 = math.radians(lat2)
    rad_lon2 = math.radians(lon2)

    try:
        pa = math.atan(rb / ra * math.tan(rad_lat1))
        pb = math.atan(rb / ra * math.tan(rad_lat2))
        temp = math.sin(pa) * math.sin(pb) + math.cos(pa) * math.cos(pb) * math.cos(rad_lon1 - rad_lon2)
        if temp > 1:
            temp = 1
        if temp < -1:
            temp = -1
        x = math.acos(temp)
        c1 = (math.sin(x) - x) * (math.sin(pa) + math.sin(pb)) ** 2 / math.cos(x / 2) ** 2
        c2 = (math.sin(x) + x) * (math.sin(pa) - math.sin(pb)) ** 2 / math.sin(x / 2) ** 2
        dr = flatten / 8 * (c1 - c2)
        distance = ra * (x + dr)
    except ZeroDivisionError:  # 有可能除数太小被认为是0
        return -1
    else:
        return distance


def get_coordinate(x1, y1, grid):
    """
    :param x1: 目标点经度
    :param y1: 目标点纬度
    :param grid: 网格颗粒度
    :return: 转化后的横纵坐标，对应 经度 纬度
    """
    x0 = 103.94
    y0 = 30.78
    distance_x = get_distance(y0, x0, y0, x1)
    distance_y = get_distance(y0, x0, y1, x0)
    if x1 < x0:
        distance_x = -distance_x
    if y1 < y0:
        distance_y = -distance_y
    return int(distance_x // grid), int(distance_y // grid)


def graham_scan(points):
    """
    Graham扫描法计算凸包
    :param points: 点集，格式为 [(x1, y1), (x2, y2), ...]
    :return: 凸包上的点，按逆时针顺序排列
    """
    if len(points) < 3:
        return points
    
    # 找到最下方的点（如果有多个，选择最左边的）
    def find_bottom_left(points):
        min_y = float('inf')
        min_x = float('inf')
        bottom_left = None
        
        for point in points:
            x, y = point
            if y < min_y or (y == min_y and x < min_x):
                min_y = y
                min_x = x
                bottom_left = point
        
        return bottom_left
    
    # 计算叉积，用于判断转向
    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    # 计算距离，用于排序
    def distance(o, a):
        return (a[0] - o[0]) ** 2 + (a[1] - o[1]) ** 2
    
    # 找到起始点
    start = find_bottom_left(points)
    
    # 按极角排序其他点
    def polar_angle_sort(point):
        if point == start:
            return (0, 0)
        angle = math.atan2(point[1] - start[1], point[0] - start[0])
        return (angle, distance(start, point))
    
    sorted_points = sorted(points, key=polar_angle_sort)
    
    # 移除重复点
    unique_points = []
    for i, point in enumerate(sorted_points):
        if i == 0 or point != sorted_points[i-1]:
            unique_points.append(point)
    
    # 如果点数少于3，无法形成凸包
    if len(unique_points) < 3:
        return unique_points
    
    # Graham扫描
    hull = [unique_points[0], unique_points[1]]
    
    for i in range(2, len(unique_points)):
        while len(hull) > 1 and cross_product(hull[-2], hull[-1], unique_points[i]) <= 0:
            hull.pop()
        hull.append(unique_points[i])
    
    return hull


def convex_hull_iou(hull1, hull2):
    """
    计算两个凸包的交集与并集的网格数比值
    :param hull1: 第一个凸包的点集，格式为 [(grid_x1, grid_y1), (grid_x2, grid_y2), ...]
    :param hull2: 第二个凸包的点集，格式为 [(grid_x1, grid_y1), (grid_x2, grid_y2), ...]
    :return: 交集网格数 / 并集网格数
    """
    def point_in_polygon(point, polygon):
        """
        判断点是否在多边形内部（包括边界）
        使用射线法
        """
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def get_grid_bounds(hull):
        """获取凸包的网格边界"""
        if not hull:
            return None
        
        min_x = min(point[0] for point in hull)
        max_x = max(point[0] for point in hull)
        min_y = min(point[1] for point in hull)
        max_y = max(point[1] for point in hull)
        
        return min_x, max_x, min_y, max_y
    
    def get_hull_grids(hull):
        """获取凸包覆盖的网格集合"""
        if not hull:
            return set()
        
        bounds = get_grid_bounds(hull)
        if not bounds:
            return set()
        
        grid_min_x, grid_max_x, grid_min_y, grid_max_y = bounds
        grids = set()
        
        # 遍历所有可能的网格
        for grid_x in range(grid_min_x, grid_max_x + 1):
            for grid_y in range(grid_min_y, grid_max_y + 1):
                # 网格中心点就是网格坐标
                center_x = grid_x + 0.5
                center_y = grid_y + 0.5
                
                # 判断网格中心是否在凸包内
                if point_in_polygon((center_x, center_y), hull):
                    grids.add((grid_x, grid_y))
        
        return grids
    
    # 获取两个凸包的网格集合
    grids1 = get_hull_grids(hull1)
    grids2 = get_hull_grids(hull2)
    
    # 计算交集和并集
    intersection = grids1 & grids2
    union = grids1 | grids2
    
    # 避免除零错误
    if len(union) == 0:
        return 0.0
    
    return len(intersection) / len(union)


def driver_clustering_by_comfort_zone(similarity_matrix, driver_id_list):
    """
    基于舒适区重合度的司机聚类算法
    
    :param similarity_matrix: 司机两两之间的相似度矩阵（IoU值）
    :param driver_id_list: 司机ID列表
    :return: 最终的司机类别集合，每个类别是一个司机ID集合
    """
    # 1. 初始化
    # 为每个司机创建一个独立的类别
    clusters = [{driver_id} for driver_id in driver_id_list]
    set_threshold = len(driver_id_list) // 10
    
    # 维护映射 f: 司机ID -> 当前所属的类别
    driver_to_cluster = {driver_id: cluster for driver_id, cluster in zip(driver_id_list, clusters)}
    
    # 当前类别集合 - 使用列表而不是集合，因为集合不可哈希
    current_clusters = clusters.copy()
    
    # 2. 计算并排序重合度
    # 生成所有司机对的重合度列表
    overlap_list = []
    for i in range(len(driver_id_list)):
        for j in range(i + 1, len(driver_id_list)):
            driver_i = driver_id_list[i]
            driver_j = driver_id_list[j]
            # 从相似度矩阵中获取IoU值
            iou = similarity_matrix[i][j] if i < len(similarity_matrix) else similarity_matrix[j][i]
            overlap_list.append((iou, driver_i, driver_j))
    
    # 按重合度从大到小排序
    overlap_list.sort(key=lambda x: x[0], reverse=True)

    # 3. 迭代合并
    for iou, driver_i, driver_j in tqdm.tqdm(overlap_list, desc='Merge clusters'):
        if iou < 0.1:
            break
        # 获取司机当前所属的类别
        cluster_i = driver_to_cluster[driver_i]
        cluster_j = driver_to_cluster[driver_j]
        
        # 检查合并条件
        if (cluster_i != cluster_j and  # 不在同一个类别
            len(cluster_i) + len(cluster_j) < set_threshold + 10):
            # (len(cluster_i) == 1 or len(cluster_j) == 1)):  # 至少有一个类别只有一个司机
            
            # 执行合并操作
            # 创建新的合并类别
            new_cluster = cluster_i.union(cluster_j)
            
            # 更新映射 f
            for driver_id in new_cluster:
                driver_to_cluster[driver_id] = new_cluster
            
            # 更新类别集合
            current_clusters.remove(cluster_i)
            current_clusters.remove(cluster_j)
            current_clusters.append(new_cluster)

    # 4. 输出最终的司机类别集合
    for driver_set in current_clusters:
        print(len(driver_set))
    return current_clusters


def get_cache_filepath(base_path, driver_count, minx, miny, maxx, maxy, file_type):
    """
    生成缓存文件路径
    
    :param base_path: 基础路径
    :param driver_count: 司机数量
    :param minx, miny, maxx, maxy: 边界信息
    :param file_type: 文件类型 ('similarity' 或 'clusters' 或 'augmented_data')
    :return: 完整的文件路径
    """
    os.makedirs(base_path, exist_ok=True)
    
    if file_type == 'similarity':
        filename = f"similarity_matrix_{driver_count}_{minx}_{miny}_{maxx}_{maxy}.pkl"
    elif file_type == 'clusters':
        filename = f"driver_clusters_{driver_count}_{minx}_{miny}_{maxx}_{maxy}.pkl"
    elif file_type == 'augmented_data':
        filename = f"augmented_data_{driver_count}_{minx}_{miny}_{maxx}_{maxy}.pkl"
    else:
        raise ValueError("file_type must be 'similarity', 'clusters' or 'augmented_data'")
    
    return os.path.join(base_path, filename)


def save_cache(data, filepath):
    """
    保存数据到缓存文件
    
    :param data: 要保存的数据
    :param filepath: 文件路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"数据已保存到: {filepath}")


def load_cache(filepath):
    """
    从缓存文件加载数据
    
    :param filepath: 文件路径
    :return: 加载的数据，如果文件不存在返回None
    """
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"数据已从缓存加载: {filepath}")
        return data
    return None


def calculate_cluster_region_boundary(driver_clusters, driver_convex_hull, driver_id_list, frequency_threshold=0.5):
    """
    计算每个司机类别的区域边界
    
    :param driver_clusters: 司机聚类结果，每个元素是一个司机ID集合
    :param driver_convex_hull: 司机凸包字典 {driver_id: convex_hull_points}
    :param driver_id_list: 司机ID列表
    :param frequency_threshold: 频率阈值，默认0.5（50%）
    :return: 每个类别的区域边界 {cluster_id: boundary_points}
    """

    
    cluster_boundaries = {}
    
    for cluster_id, cluster in enumerate(driver_clusters):
        if len(cluster) < 5:
            continue
        # print(f"计算类别 {cluster_id + 1} 的区域边界...")
        
        # 获取该类别的所有司机
        cluster_drivers = list(cluster)
        cluster_size = len(cluster_drivers)
        
        if cluster_size == 0:
            continue
            
        # 计算该类别的所有凸包覆盖频率
        frequency_map = {}
        
        # 遍历该类别的每个司机
        for driver_id in cluster_drivers:
            if driver_id not in driver_convex_hull:
                continue
                
            convex_hull = driver_convex_hull[driver_id]
            
            # 获取凸包覆盖的所有网格点
            covered_grids = get_hull_grids(convex_hull)
            
            # 统计每个网格点的覆盖次数
            for grid_point in covered_grids:
                frequency_map[grid_point] = frequency_map.get(grid_point, 0) + 1
        
        # 计算频率阈值
        min_frequency = max(1, int(cluster_size * frequency_threshold))
        
        # 找出频率大于阈值的网格点
        boundary_grids = []
        for grid_point, frequency in frequency_map.items():
            if frequency >= min_frequency:
                boundary_grids.append(grid_point)
        
        # 将网格点转换为边界点
        if boundary_grids:
            # 使用凸包算法找到边界
            boundary_points = graham_scan(boundary_grids)
            cluster_boundaries[cluster_id] = boundary_points
        else:
            # 如果没有满足条件的点，使用所有司机的凸包并集
            all_points = []
            for driver_id in cluster_drivers:
                if driver_id in driver_convex_hull:
                    all_points.extend(driver_convex_hull[driver_id])
            if all_points:
                cluster_boundaries[cluster_id] = graham_scan(all_points)
            else:
                cluster_boundaries[cluster_id] = []
        
        # print(f"类别 {cluster_id + 1}: {len(cluster_drivers)} 个司机，边界点 {len(cluster_boundaries[cluster_id])} 个")
    
    return cluster_boundaries


def get_hull_grids(hull):
    """
    获取凸包覆盖的所有网格点
    
    :param hull: 凸包点集
    :return: 网格点集合
    """
    if not hull:
        return set()
    
    def point_in_polygon(point, polygon):
        """判断点是否在多边形内部（包括边界）"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    # 获取凸包的边界
    min_x = min(point[0] for point in hull)
    max_x = max(point[0] for point in hull)
    min_y = min(point[1] for point in hull)
    max_y = max(point[1] for point in hull)
    
    grids = set()
    
    # 遍历所有可能的网格
    for grid_x in range(int(min_x), int(max_x) + 1):
        for grid_y in range(int(min_y), int(max_y) + 1):
            # 计算网格中心点
            center_x = grid_x + 0.5
            center_y = grid_y + 0.5
            
            # 判断网格中心是否在凸包内
            if point_in_polygon((center_x, center_y), hull):
                grids.add((grid_x, grid_y))
    
    return grids


def visualize_cluster_regions(driver_clusters, cluster_boundaries, driver_id_list, save_path=None):
    """
    可视化司机类别的区域边界
    
    :param driver_clusters: 司机聚类结果
    :param cluster_boundaries: 每个类别的区域边界
    :param driver_id_list: 司机ID列表
    :param save_path: 保存图片的路径
    """
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 为每个类别分配不同的颜色
    colors = []
    for i in range(len(driver_clusters)):
        hue = i / len(driver_clusters)
        color = hsv_to_rgb([hue, 0.7, 0.9])
        colors.append(color)
    
    # 绘制每个类别的区域边界
    for cluster_id, boundary_points in cluster_boundaries.items():
        if len(boundary_points) < 3:
            continue
            
        # 创建多边形
        polygon = patches.Polygon(boundary_points, 
                                facecolor=colors[cluster_id], 
                                alpha=0.3, 
                                edgecolor=colors[cluster_id], 
                                linewidth=2,
                                label=f'type {cluster_id + 1} ({len(driver_clusters[cluster_id])} drivers)')
        ax.add_patch(polygon)
    
    # 设置图表属性
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('司机类别区域分布图')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 自动调整坐标轴范围
    ax.autoscale_view()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"区域分布图已保存到: {save_path}")
    
    plt.show()


def print_cluster_statistics(driver_clusters, cluster_boundaries, driver_id_list):
    """
    打印聚类统计信息
    
    :param driver_clusters: 司机聚类结果
    :param cluster_boundaries: 每个类别的区域边界
    :param driver_id_list: 司机ID列表
    """
    print("\n=== 司机聚类统计信息 ===")
    print(f"总司机数: {len(driver_id_list)}")
    print(f"聚类数量: {len(driver_clusters)}")
    
    for cluster_id, cluster in enumerate(driver_clusters):
        cluster_size = len(cluster)
        boundary_points = cluster_boundaries.get(cluster_id, [])
        
        print(f"\n类别 {cluster_id + 1}:")
        print(f"  司机数量: {cluster_size}")
        print(f"  边界点数: {len(boundary_points)}")
        print(f"  司机ID: {sorted(cluster)}")
        
        if boundary_points:
            # 计算区域面积（网格数）
            area_grids = len(get_hull_grids(boundary_points))
            print(f"  区域面积: {area_grids} 个网格")


def visualize_convex_hull(hull_points):
    """
    可视化单个凸包的范围
    
    :param hull_points: 凸包点集，格式为 [(x1, y1), (x2, y2), ...]
    """
    
    if len(hull_points) < 3:
        print("错误：凸包点数不足（需要至少3个点）")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制凸包
    polygon = patches.Polygon(hull_points, 
                            facecolor='lightblue', 
                            alpha=0.3, 
                            edgecolor='blue', 
                            linewidth=2,
                            label='Convex Hull')
    ax.add_patch(polygon)
    
    # 绘制凸包顶点
    x_coords = [p[0] for p in hull_points]
    y_coords = [p[1] for p in hull_points]
    ax.scatter(x_coords, y_coords, c='red', s=100, alpha=0.8, zorder=5, label='Vertices')
    
    # 绘制凸包中心点
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    ax.scatter(center_x, center_y, c='green', s=150, alpha=0.8, zorder=6, 
              marker='*', label='Center')
    
    # 设置图表属性
    ax.set_xlabel('Grid X Coordinate')
    ax.set_ylabel('Grid Y Coordinate')
    ax.set_title('Convex Hull Visualization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 自动调整坐标轴范围
    ax.autoscale_view()
    
    # 添加统计信息文本
    area_grids = len(get_hull_grids(hull_points))
    info_text = f'Vertices: {len(hull_points)}\nArea: {area_grids} grids'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.show()


def visualize_two_convex_hulls(hull_points_1, hull_points_2, label_1="Hull 1", label_2="Hull 2"):
    """
    将两个凸包画到一张图上
    
    :param hull_points_1: 第一个凸包点集，格式为 [(x1, y1), (x2, y2), ...]
    :param hull_points_2: 第二个凸包点集，格式为 [(x1, y1), (x2, y2), ...]
    :param label_1: 第一个凸包的标签
    :param label_2: 第二个凸包的标签
    """
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = ['lightblue', 'lightcoral']
    edge_colors = ['blue', 'red']
    center_colors = ['green', 'orange']
    
    hulls = [hull_points_1, hull_points_2]
    labels = [label_1, label_2]
    
    for i, (hull_points, label) in enumerate(zip(hulls, labels)):
        if len(hull_points) < 3:
            print(f"错误：{label} 凸包点数不足（需要至少3个点）")
            continue
        
        # 绘制凸包
        polygon = patches.Polygon(hull_points, 
                                facecolor=colors[i], 
                                alpha=0.3, 
                                edgecolor=edge_colors[i], 
                                linewidth=2,
                                label=f'{label} Convex Hull')
        ax.add_patch(polygon)
        
        # 绘制凸包顶点
        x_coords = [p[0] for p in hull_points]
        y_coords = [p[1] for p in hull_points]
        ax.scatter(x_coords, y_coords, c=edge_colors[i], s=80, alpha=0.8, zorder=5, 
                  label=f'{label} Vertices')
        
        # 绘制凸包中心点
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        ax.scatter(center_x, center_y, c=center_colors[i], s=120, alpha=0.8, zorder=6, 
                  marker='*', label=f'{label} Center')
        
        # 添加统计信息文本
        area_grids = len(get_hull_grids(hull_points))
        info_text = f'{label}: {len(hull_points)} vertices, {area_grids} grids'
        ax.text(0.02, 0.95 - i*0.05, info_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 设置图表属性
    ax.set_xlabel('Grid X Coordinate')
    ax.set_ylabel('Grid Y Coordinate')
    ax.set_title('Two Convex Hulls Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 自动调整坐标轴范围
    ax.autoscale_view()
    
    plt.show()


def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside
