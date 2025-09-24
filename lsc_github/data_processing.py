import tqdm
import datetime
import pandas as pd
import os
import pickle
from utils import get_coordinate, graham_scan, convex_hull_iou, driver_clustering_by_comfort_zone, get_cache_filepath, save_cache, load_cache
from utils import calculate_cluster_region_boundary, visualize_cluster_regions, visualize_convex_hull, visualize_two_convex_hulls, point_in_polygon


def poi_processing(poi_path, grid_granularity):
    """
    Load POI
    :param poi_path:
    :param grid_granularity:
    :return: {(x,y): [(poi_type, num), ...]}, [min_lon, min_lat, max_lon, max_lat]
    """
    poi_data = {}
    minx, miny, maxx, maxy = 1000, 1000, -1000, -1000
    poi_table = ['生活服务', '教育培训', '交通设施', '汽车服务', '道路', '休闲娱乐', '文化传媒', '丽人', '房地产',
                 '美食', '酒店', '公司企业', '购物', '政府机构', '金融', '医疗', '旅游景点', '运动健身', '自然地物',
                 '铁路', '公交线路']
    poi_kind_table = ['住宅区', '公司', '路口', '家电数码', '其他', '农家院', '科研机构', '家居建材', '居民委员会',
                      '宿舍', '停车场', '投资理财', '物流公司', '商铺', '药店', '内部楼栋', '文化宫', '幼儿园',
                      '殡葬服务', '文物古迹', '汽车维修', '美发', '厂矿', '中餐厅', '各级政府', 'ktv', '小学',
                      '小吃快餐店', '培训机构', '汽车美容', '休闲广场', '公园', '高等院校', '公用事业', '加油加气站',
                      '超市', '收费站', '公共厕所', '农林园艺', '中学', '照相馆', '茶座', '艺术团体', '景点', '便利店',
                      '公检法机构', '汽车配件', '路侧停车位', '体育场馆', '风景区', '行政单位', '写字楼',
                      '房产中介机构', '桥', '星级酒店', '蛋糕甜品店', '美容', '汽车销售', '党派团体', '市场',
                      '社会团体', '洗浴按摩', '外国餐厅', '洗衣店', '快捷酒店', '通讯营业厅', '充电站', '火车站',
                      '园区', '诊所', '急救中心', '售票处', '美体', '宠物服务', '汽车检测场', '银行', '汽车租赁',
                      '公寓式酒店', '专科医院', '网吧', '福利机构', '维修点', 'atm', '家政服务', '展览馆', '咖啡厅',
                      '信用社', '报刊亭', '山峰', '高速公路', '港口', '度假村', '公交车站', '图文快印店', '水系',
                      '政治教育机构', '博物馆', '彩票销售点', '长途汽车站', '集市', '美术馆', '综合医院', '教堂',
                      '电影院', '特殊教育学校', '疗养院', '购物中心', '动物园', '疾控中心', '县道', '飞机场',
                      '健身中心', '游戏场所', '广播电视', '成人教育', '植物园', '城市快速路', '游乐园', '美甲',
                      '图书馆', '酒吧', '亲子教育']

    df = pd.read_table(poi_path, header=None)
    for i in tqdm.tqdm(range(df.shape[0]), desc='Load POI'):
        site_str = df.iloc[i, 0]
        poi_str = df.loc[i, 1]
        if isinstance(poi_str, float) or isinstance(site_str, float):  # skip nan value
            continue

        # get coordinate
        site_list = site_str.split(',')
        lat = float(site_list[1])
        lon = float(site_list[2])
        x, y = get_coordinate(lon, lat, grid_granularity)
        minx, miny, maxx, maxy = min(minx, x), min(miny, y), max(maxx, x), max(maxy, y)
        if (x, y) in poi_data:
            continue

        # construct poi data: {(x,y): [(poi_type, num), ..., ()]}
        poi_list = poi_str.split('|')
        for poi in poi_list:
            if ':' in poi:
                poi_data.setdefault((x, y), []).append((str(poi.split(':')[0]), int(poi.split(':')[1])))
            if ';' in poi:
                poi_data[(x, y)].append((poi.split(';')[0], poi.split(';')[1]))

        # get poi feature
        feature_tmp = [0 for _ in range(len(poi_table))]
        for name, num in poi_data[(x, y)][:-1]:
            feature_tmp[poi_table.index(name)] = num
        if isinstance(poi_data[(x, y)][-1][1], str):
            feature_tmp.append(poi_kind_table.index(poi_data[(x, y)][-1][1]) + 1)
        else:
            feature_tmp.append(0)
        poi_data[(x, y)] = feature_tmp

    print('POI grid map boundary:', minx, miny, maxx, maxy)
    return poi_data, [minx, miny, maxx, maxy]


def driver_order_processing(driver_order_path, grid_granularity, poi_boundary):
    """
    :param poi_boundary:
    :param driver_order_path:
    :param grid_granularity: {driver_id:[d_id, t, p, d], ...}, [min_lon, min_lat, max_lon, max_lat]
    :return:
    """

    # Load driver_order
    driver_data = {}
    driver_convex_hull = {}
    driver_id_list = []
    similarity_matrix = []
    minx, miny, maxx, maxy = 1000, 1000, -1000, -1000
    with open(driver_order_path, 'r', encoding='utf8') as f:
        for idx, line in tqdm.tqdm(enumerate(f, 1), desc='Load driver order'):
            data = line.split(',')
            data = [float(x) if '.' in x and len(x) <= 11 else x for x in data]
            data[4], data[5] = get_coordinate(data[4], data[5], grid_granularity)
            minx, miny, maxx, maxy = min(minx, data[4]), min(miny, data[5]), max(maxx, data[4]), max(maxy, data[5])
            data[6], data[7] = get_coordinate(data[6], data[7], grid_granularity)
            minx, miny, maxx, maxy = min(minx, data[6]), min(miny, data[7]), max(maxx, data[6]), max(maxy, data[7])
            driver_data.setdefault(data[1], []).append(data)
            if idx > 10000:
                break

    # Boundary filtering
    delete_driver = []
    for driver_id, order_data in driver_data.items():
        for order in order_data:
            if order[4] < poi_boundary[0] or order[4] > poi_boundary[2] or \
               order[5] < poi_boundary[1] or order[5] > poi_boundary[3] or \
               order[6] < poi_boundary[0] or order[6] > poi_boundary[2] or \
               order[7] < poi_boundary[1] or order[7] > poi_boundary[3]:
                delete_driver.append(driver_id)
                break
    for driver_id in delete_driver:
        del driver_data[driver_id]
    print('Number of cross-border drivers:', len(delete_driver))

    # Logical conflict in order time filtering
    delete_driver = []
    for driver_id, order_data in driver_data.items():
        if len(order_data) <= 2:
            delete_driver.append(driver_id)
            continue
        sorted_order_data = sorted(order_data, key=lambda x: int(x[2]))
        drop_off_hour = sorted_order_data[0][3]
        for order in sorted_order_data[1:]:
            if order[2] < drop_off_hour:
                delete_driver.append(driver_id)
                break
            drop_off_hour = order[3]
    for driver_id in delete_driver:
        del driver_data[driver_id]
    print('Number of logical conflicting in order time drivers:', len(delete_driver))

    # print('Driver order grid map boundary:', minx, miny, maxx, maxy)
    print('Total number of remaining drivers:', len(driver_data))

    # calculate driver convex hull
    for driver_id, order_data in driver_data.items():
        driver_id_list.append(driver_id)
        order_data = sorted(order_data, key=lambda x: int(x[2]))
        points = set()
        for order in order_data:
            points.add((order[4], order[5]))  # pickup point
            points.add((order[6], order[7]))  # dropoff point
        points = list(points)
        convex_hull = graham_scan(points)
        driver_convex_hull[driver_id] = convex_hull
    # visualize_convex_hull(driver_convex_hull['bgkifm97f8tt2fxBhijcglhajazC4eCy'])
    # visualize_two_convex_hulls(driver_convex_hull['6hid5a9f95zAadyybaceak9dc8vFcczr'], driver_convex_hull['jelceilgi.Bu0kAm6hge7gf7d8tD5nDq'])

    # load similarity matrix
    similarity_matrix_path = r"E:\Project\traffic\lsc\data\similarity_matrix"
    similarity_filepath = get_cache_filepath(similarity_matrix_path, len(driver_id_list), minx, miny, maxx, maxy, 'similarity')
    similarity_matrix = load_cache(similarity_filepath)
    
    if similarity_matrix is None:
        similarity_matrix = []
        print("计算相似度矩阵...")
        # calculate similarity matrix
        for i in tqdm.tqdm(range(len(driver_convex_hull) - 1), desc='Calculate similarity'):
            convex_hull_u = driver_convex_hull[driver_id_list[i]]
            similarity_u = [0.0 for _ in range(i + 1)]
            for j in range(i + 1, len(driver_convex_hull)):
                convex_hull_v = driver_convex_hull[driver_id_list[j]]
                iou = convex_hull_iou(convex_hull_u, convex_hull_v)
                similarity_u.append(iou)
            similarity_matrix.append(similarity_u)
        save_cache(similarity_matrix, similarity_filepath)
    
    driver_clusters = driver_clustering_by_comfort_zone(similarity_matrix, driver_id_list)
    
    # 计算每个类别的区域边界
    cluster_boundaries = calculate_cluster_region_boundary(driver_clusters, driver_convex_hull, driver_id_list, 0.8)
    
    # 可视化区域分布
    # try:
    #     save_path = r"E:\Project\traffic\lsc\data\cluster_regions.png"
    #     visualize_cluster_regions(driver_clusters, cluster_boundaries, driver_id_list, save_path)
    # except Exception as e:
    #     print(f"可视化失败: {e}")
    
    return driver_data, [minx, miny, maxx, maxy], driver_clusters, cluster_boundaries


def add_driver_id_to_cluster(driver_data, driver_clusters, cluster_boundaries, driver_order_path, grid_granularity, poi_boundary):
    """
    读取未处理的司机，并根据其舒适区（凸包）与现有大聚类区域的相似度，将其分配到最相似的聚类中。
    :param driver_data: 已处理的司机数据
    :param driver_clusters: 司机聚类结果
    :param cluster_boundaries: 大聚类的区域边界
    :param driver_order_path: 司机订单文件路径
    :param grid_granularity: 网格粒度
    :param poi_boundary: POI边界
    :return: 更新后的 driver_data, driver_clusters
    """
    existing_driver_ids = set(driver_data.keys())
    print("Reading all driver orders to find new drivers...")
    new_drivers_from_file = {}
    with open(driver_order_path, 'r', encoding='utf8') as f:
        for line in tqdm.tqdm(f, desc='Scanning all driver orders'):
            data = line.split(',')
            data = [float(x) if '.' in x and len(x) <= 11 else x for x in data]
            driver_id = data[1]
            if driver_id in existing_driver_ids:
                continue
            
            data[4], data[5] = get_coordinate(data[4], data[5], grid_granularity)
            data[6], data[7] = get_coordinate(data[6], data[7], grid_granularity)
            new_drivers_from_file.setdefault(driver_id, []).append(data)

    print(f"Found {len(new_drivers_from_file)} new potential drivers.")

    # Filter new drivers
    # Boundary filtering
    delete_driver = []
    for driver_id, order_data in new_drivers_from_file.items():
        for order in order_data:
            if order[4] < poi_boundary[0] or order[4] > poi_boundary[2] or \
               order[5] < poi_boundary[1] or order[5] > poi_boundary[3] or \
               order[6] < poi_boundary[0] or order[6] > poi_boundary[2] or \
               order[7] < poi_boundary[1] or order[7] > poi_boundary[3]:
                delete_driver.append(driver_id)
                break
    for driver_id in delete_driver:
        del new_drivers_from_file[driver_id]

    # Logical conflict in order time filtering
    delete_driver = []
    for driver_id, order_data in new_drivers_from_file.items():
        if len(order_data) <= 2:
            delete_driver.append(driver_id)
            continue
        sorted_order_data = sorted(order_data, key=lambda x: int(x[2]))
        drop_off_hour = sorted_order_data[0][3]
        for order in sorted_order_data[1:]:
            if order[2] < drop_off_hour:
                delete_driver.append(driver_id)
                break
            drop_off_hour = order[3]
    for driver_id in delete_driver:
        del new_drivers_from_file[driver_id]

    new_driver_data = new_drivers_from_file
    print(f"Number of new drivers after filtering: {len(new_driver_data)}")

    # Calculate convex hulls for new drivers
    new_driver_convex_hulls = {}
    for driver_id, order_data in new_driver_data.items():
        points = set()
        for order in order_data:
            points.add((order[4], order[5]))
            points.add((order[6], order[7]))
        points = list(points)
        convex_hull = graham_scan(points)
        if convex_hull:
            new_driver_convex_hulls[driver_id] = convex_hull

    # Assign new drivers to the most similar cluster
    assigned_count = 0
    for driver_id, hull in tqdm.tqdm(new_driver_convex_hulls.items(), desc="Assigning new drivers to clusters"):
        best_cluster_id = -1
        max_iou = -1.0
        
        for cluster_id, boundary_hull in cluster_boundaries.items():
            if not boundary_hull:
                continue
            
            iou = convex_hull_iou(hull, boundary_hull)
            if iou > max_iou:
                max_iou = iou
                best_cluster_id = cluster_id
                
        if best_cluster_id != -1 and max_iou > 0:
            driver_clusters[best_cluster_id].add(driver_id)
            driver_data[driver_id] = new_driver_data[driver_id]
            assigned_count += 1
            
    print(f"Assigned {assigned_count} new drivers to existing large clusters.")
    return driver_data, driver_clusters


def ground_truth_processing(ground_truth_path, grid_granularity, poi_boundary):
    # load ground_truth
    ground_truth_data = []
    label = []
    ground_truth_pd = pd.read_excel(ground_truth_path)
    for i in tqdm.tqdm(range(ground_truth_pd.shape[0]), desc='Load ground truth'):
        x, y = get_coordinate(ground_truth_pd.iloc[i, 3], ground_truth_pd.iloc[i, 2], grid_granularity)
        ground_truth_data.append((x, y))

    for i in range(poi_boundary[2] - poi_boundary[0] + 1):
        for j in range(poi_boundary[3] - poi_boundary[1] + 1):
            if (i + poi_boundary[0], j + poi_boundary[1]) in ground_truth_data:
                label.append(1)
            else:
                label.append(0)
    return label


def build_features(pois, driver_orders, dimension_information):
    origin_features = {}    # dim=[24, ?, 3 * k]
    features = []
    grid_feature_max_length = [0 for _ in range(24)]
    grid_num = [0 for _ in range(24)]
    lon_offset = dimension_information[2]
    lat_offset = dimension_information[3]
    lat_num = dimension_information[1]

    for driver_id, driver_order in driver_orders.items():
        sorted_driver_order = sorted(driver_order, key=lambda x: int(x[2]))
        now_driver = []

        # prepare grid_id time
        for order in sorted_driver_order:
            pick_up_grid_id = (order[4] + lon_offset) * lat_num + order[5] + lat_offset
            drop_off_grid_id = (order[6] + lon_offset) * lat_num + order[7] + lat_offset
            pick_up_hour = datetime.datetime.fromtimestamp(int(order[2])).hour
            drop_off_hour = datetime.datetime.fromtimestamp(int(order[3])).hour
            now_driver.append((pick_up_hour, drop_off_hour, pick_up_grid_id, drop_off_grid_id))

        # build feature     last_drop_off -> now_pick_up
        for i in range(1, len(now_driver)):
            pick_time = now_driver[i][0]
            drop_time = now_driver[i - 1][1]
            offset = now_driver[i][2] - now_driver[i - 1][3]
            origin_features.setdefault(str(pick_time), {}).setdefault(str(now_driver[i - 1][3]), []).append(
                (pick_time, now_driver[i][2]))
                # (pick_time - drop_time, offset)
    
    for i in range(24):
        if str(i) not in origin_features:
            origin_features[str(i)] = {}

    for i in range(24):
        grid_num[i] = len(origin_features[str(i)])
        for k, v in origin_features[str(i)].items():
            if len(v) > grid_feature_max_length[i]:
                grid_feature_max_length[i] = len(v)
    
    print('Grid num:', grid_num)
    print('Grid feature max length:', grid_feature_max_length)

    # Build final features
    layer_dimension = [x * 2 // 10 * 10 if x % 10 == 0 else (x * 2 // 10 + 1) * 10 for x in grid_feature_max_length]
    layer_dimension = [10, 10, 10, 10, 20, 20, 20, 30, 40, 40, 40, 40, 50, 50, 60, 40, 40, 40, 40, 40, 30, 30, 30, 20]  
    print('Layer dimension:', layer_dimension)
    # add poi feature
    for i in range(dimension_information[0]):
        for j in range(dimension_information[1]):
            if (i - lon_offset, j - lat_offset) in pois:
                features.append(pois[(i - lon_offset, j - lat_offset)])
            else:
                features.append([0 for _ in range(22)])
    # add driver order
    for layer in tqdm.tqdm(range(24), desc='Add driver features'):
        for grid_id in range(dimension_information[0] * dimension_information[1]):
            if str(grid_id) in origin_features[str(layer)]:
                for element in origin_features[str(layer)][str(grid_id)]:
                    features[grid_id].extend([element[0], element[1]])
                features[grid_id].extend((layer_dimension[layer] - 2 *
                                          len(origin_features[str(layer)][str(grid_id)])) * [0])
            else:
                features[grid_id].extend([0 for _ in range(layer_dimension[layer])])
    return features


def build_adjs(driver_orders, dimension_information, sparse):
    from collections import defaultdict
    import math
    def boundary_judge(x, y):
        if 0 <= x < dimension_information[0] and 0 <= y < dimension_information[1]:
            return True
        return False

    node_num = dimension_information[0] * dimension_information[1]
    adjs = defaultdict(dict)  # 邻接表
    move = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Connect neighbors
    for i in tqdm.tqdm(range(dimension_information[0]), desc='Adjs connect neighbors'):
        for j in range(dimension_information[1]):
            u_id = i * dimension_information[1] + j
            for now_move in move:
                ni, nj = i + now_move[0], j + now_move[1]
                if boundary_judge(ni, nj):
                    v_id = ni * dimension_information[1] + nj
                    adjs[u_id][v_id] = adjs[u_id].get(v_id, 0) + 1

    # Connect driver order pick_up and drop_off
    for driver_id, driver_order in tqdm.tqdm(driver_orders.items(), desc='Adjs add driver orders'):
        sorted_driver_order = sorted(driver_order, key=lambda x: int(x[2]))
        for order in sorted_driver_order:
            u_id = (order[4] + dimension_information[2]) * dimension_information[1] + \
                order[5] + dimension_information[3]
            v_id = (order[6] + dimension_information[2]) * dimension_information[1] + \
                order[7] + dimension_information[3]
            adjs[u_id][v_id] = adjs[u_id].get(v_id, 0) + 1

    return dict(adjs)


def filter_labels_by_cluster_boundaries(ground_truth_data, cluster_boundaries, dimension_information, min_positive_samples=300):
    """
    根据聚类边界过滤标签，如果凸包内正样本不足，从最近样本补充
    
    :param ground_truth_data: 原始标签数据
    :param cluster_boundaries: 聚类边界字典
    :param dimension_information: 维度信息 [n, m, x_offset, y_offset]
    :param min_positive_samples: 最小正样本数量，默认300
    :return: 每个聚类的标签字典
    """
    n, m = dimension_information[0], dimension_information[1]
    labels_by_cluster = {}
    
    for cluster_id, boundary in cluster_boundaries.items():
        label = [0] * (n * m)
        
        if not boundary or len(boundary) < 3:
            labels_by_cluster[cluster_id] = label
            continue
        
        # 第一步：标记凸包内的点
        hull_points = set()
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                if point_in_polygon((i, j), boundary):
                    label[idx] = ground_truth_data[idx]
                    hull_points.add((i, j))
        
        # 统计凸包内的正样本数量
        positive_in_hull = sum(1 for idx in range(n * m) if label[idx] == 1)
        print(f"Cluster {cluster_id}: 凸包内正样本数量 = {positive_in_hull}")
        
        # 如果正样本不足，从最近样本补充
        if positive_in_hull < min_positive_samples:
            print(f"Cluster {cluster_id}: 正样本不足{min_positive_samples}个，开始补充...")
            
            # 计算所有点到凸包边界的距离
            distances = []
            for i in range(n):
                for j in range(m):
                    if (i, j) not in hull_points:  # 只考虑凸包外的点
                        # 计算点到凸包边界的最短距离
                        min_distance = float('inf')
                        for k in range(len(boundary)):
                            p1 = boundary[k]
                            p2 = boundary[(k + 1) % len(boundary)]
                            dist = point_to_line_segment_distance((i, j), p1, p2)
                            min_distance = min(min_distance, dist)
                        
                        if ground_truth_data[i * m + j] == 1:  # 只考虑正样本
                            distances.append((min_distance, i, j))
            
            # 按距离排序，选择最近的样本
            distances.sort(key=lambda x: x[0])
            
            # 补充正样本直到达到目标数量
            samples_needed = min_positive_samples - positive_in_hull
            samples_added = 0
            
            for dist, i, j in distances:
                if samples_added >= samples_needed:
                    break
                idx = i * m + j
                if label[idx] == 0:  # 确保还没有被标记
                    label[idx] = 1
                    samples_added += 1
            
            print(f"Cluster {cluster_id}: 补充了 {samples_added} 个正样本")
        
        labels_by_cluster[cluster_id] = label
    
    return labels_by_cluster


def point_to_line_segment_distance(point, line_start, line_end):
    """
    计算点到线段的最短距离
    
    :param point: 点坐标 (x, y)
    :param line_start: 线段起点 (x, y)
    :param line_end: 线段终点 (x, y)
    :return: 最短距离
    """
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # 线段长度
    line_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    
    if line_length == 0:
        # 线段退化为点
        return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
    
    # 计算投影参数 t
    t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_length ** 2)))
    
    # 投影点坐标
    projection_x = x1 + t * (x2 - x1)
    projection_y = y1 + t * (y2 - y1)
    
    # 点到投影点的距离
    distance = ((px - projection_x) ** 2 + (py - projection_y) ** 2) ** 0.5
    
    return distance


def data_prepare(poi_path, driver_order_path, ground_truth_path, grid_granularity):
    cache_path = "E:/Project/traffic/lsc/data/cluster_data_cache_v2.pkl"
    cached = load_cache(cache_path)
    if cached:
        print("Loaded cluster data from cache.")
        return cached
    print('Loading data...')

    poi_data, poi_boundary = poi_processing(poi_path, grid_granularity)
    poi_boundary = [-28, -75, 73, 29]
    driver_data, _, driver_clusters, cluster_boundaries = driver_order_processing(driver_order_path, grid_granularity, poi_boundary)

    # Caching for augmented driver data and clusters
    augmented_data_path = r"E:\Project\traffic\lsc\data\augmented_data_v2"
    augmented_filepath = get_cache_filepath(augmented_data_path, len(driver_data),
                                            poi_boundary[0], poi_boundary[1],
                                            poi_boundary[2], poi_boundary[3],
                                            'augmented_data')
    cached_data = load_cache(augmented_filepath)
    if cached_data:
        print("Loading augmented driver data from cache.")
        driver_data, driver_clusters = cached_data
    else:
        print("Augmented data cache not found. Running the time-consuming process...")
        driver_data, driver_clusters = add_driver_id_to_cluster(driver_data, driver_clusters, cluster_boundaries, driver_order_path, grid_granularity, poi_boundary)
        save_cache((driver_data, driver_clusters), augmented_filepath)
    
    # del_driver_cluster_set = []
    # for driver_id_set in driver_clusters:
    #     if len(driver_id_set) < 10:
    #         del_driver_cluster_set.append(driver_id_set)
    # for driver_id_set in del_driver_cluster_set:
    #     driver_clusters.remove(driver_id_set)

    ground_truth_data = ground_truth_processing(ground_truth_path, grid_granularity, poi_boundary)

    # final_boundary = [min(poi_boundary[0], driver_boundary[0]), min(poi_boundary[1], driver_boundary[1]),
    #                   max(poi_boundary[2], driver_boundary[2]), max(poi_boundary[3], driver_boundary[3])]
    final_boundary = poi_boundary
    n = final_boundary[2] + abs(final_boundary[0]) + 1
    m = final_boundary[3] + abs(final_boundary[1]) + 1
    x_offset = abs(final_boundary[0])
    y_offset = abs(final_boundary[1])
    dimension_information = [n, m, x_offset, y_offset]

    print('Final grid map boundary:', final_boundary[0], final_boundary[1], final_boundary[2], final_boundary[3])
    print('Dimension_information:', dimension_information)
    print('Finish loading data\n')

    print('Building features for each cluster')
    features_by_cluster = {}
    adjs_by_cluster = {}
    for cluster_id in cluster_boundaries.keys():
        driver_id_set = driver_clusters[cluster_id]
        driver_data_subset = {driver_id: driver_data[driver_id] for driver_id in driver_id_set if driver_id in driver_data}
        
        features = build_features(poi_data, driver_data_subset, dimension_information)
        features_by_cluster[cluster_id] = features

        adjs = build_adjs(driver_data_subset, dimension_information, True)
        adjs_by_cluster[cluster_id] = adjs
    
    print('Finish building features')

    labels_by_cluster = filter_labels_by_cluster_boundaries(ground_truth_data, cluster_boundaries, dimension_information)

    save_cache((features_by_cluster, adjs_by_cluster, labels_by_cluster, driver_clusters, cluster_boundaries), cache_path)
    return features_by_cluster, adjs_by_cluster, labels_by_cluster, driver_clusters, cluster_boundaries
