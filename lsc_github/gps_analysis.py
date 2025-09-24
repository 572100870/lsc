import datetime
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import tqdm
import json
import os
import folium
import random
import pandas as pd
from folium.plugins import HeatMap
matplotlib.use('TkAgg')


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
    :return: 转化后的横纵坐标
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


def single_driver():
    last = []
    x_coords = []
    y_coords = []
    times = []
    trajectory_idx_list = []
    taxi_id = ['co85eh7fc1wr8jxsioej4e7ga9Et.nCq',
               '49a75bbhj.zr2fzq4jhe4gh9b0zq7nxo',
               'dnbffmcghbtF0hBwidbf6k8ia-sD-cyB'][2]
    with open('D:/Project/traffic/gps_data01-03/gps_20161101', 'r') as f:
        # 取数据
        for idx, line in enumerate(f, 1):
            data = line.split(',')
            if data[0] != taxi_id and not last:     # 跳过非目标司机
                continue
            if data[0] != taxi_id:  # 目标司机结束
                print(data[0])
                times.append(str(last[1][2].hour) + ':' + str(last[1][2].minute))
                print(last[1])
                break
            x, y = get_coordinate(float(data[3]), float(data[4]), 50)   # 获取横纵坐标
            data.append(data[2])    # 保留原始时间戳
            data[2] = datetime.datetime.fromtimestamp(int(data[2]))     # 转化时间戳
            data = data + [x, y]
            if not last:
                x_coords.append([x])
                y_coords.append([y])
                last = [data[1], data]  # 记录轨迹的起始点
                trajectory_idx_list.append((data[5], len(trajectory_idx_list)))     # 保留时间戳和序号用于对时间排序
                times.append(str(data[2].hour) + ':' + str(data[2].minute))     # 记录起止时间
                print(data)
            if last[0] != data[1]:
                x_coords.append([x])
                y_coords.append([y])
                trajectory_idx_list.append((data[5], len(trajectory_idx_list)))
                times.append(str(last[1][2].hour) + ':' + str(last[1][2].minute))
                print(last[1])
                times.append(str(data[2].hour) + ':' + str(data[2].minute))
                print(data)
            x_coords[-1].append(x)
            y_coords[-1].append(y)
            last = [data[1], data]

        sorted_trajectory_idx_list = sorted(trajectory_idx_list, key=lambda x: x[0])
        # 绘图
        plt.figure(figsize=(10, 6))
        for i in range(len(sorted_trajectory_idx_list)):
            real_i = i
            i = sorted_trajectory_idx_list[i][1]
            plt.plot(x_coords[i], y_coords[i], marker='o', linestyle='-', linewidth=0.5, markersize=1.5)
                     # label=f'Trajectory {real_i + 1} ' + str(times[i * 2]) + ' - ' + str(times[i * 2 + 1]))  # 轨迹
            plt.scatter(x_coords[i][0], y_coords[i][0], color='green', marker='^', s=100,
                        label='Start' if i == 0 else "")    # 绘制起点
            plt.scatter(x_coords[i][-1], y_coords[i][-1], color='red', marker='*', s=100,
                        label='End' if i == 0 else "")      # 绘制终点
            middle_index = len(x_coords[i]) // 2
            # plt.text(x_coords[i][middle_index], y_coords[i][middle_index], f'Trajectory {real_i + 1}',
            #          color='black', fontsize=8, ha='center')    # 加上轨迹编号

        plt.title("Trajectory Plot")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        plt.xticks(np.arange(np.floor(xmin), np.ceil(xmax) + 1, 5))
        plt.yticks(np.arange(np.floor(ymin), np.ceil(ymax) + 1, 5))
        plt.legend()
        plt.grid(True)
        plt.show()


def road_network():
    """
    绘制完整路网
    :return:
    """
    flag = 2
    last = []
    x_coords = []
    y_coords = []
    points = {}
    time_name = '20161101'
    if os.path.exists('D:/Project/traffic/gps_data01-03/road_network_' + time_name + '.jsonl'):
        with open('D:/Project/traffic/gps_data01-03/road_network_' + time_name + '.jsonl', 'r', encoding='utf-8') as f:
            data = json.load(f)
            x_coords = data['x']
            y_coords = data['y']
    else:
        with open('D:/Project/traffic/gps_data01-03/gps_' + time_name, 'r') as f:
            for idx, line in tqdm.tqdm(enumerate(f, 1), desc='Processing lines'):
                if idx % 10000000 == 0:
                    print(len(x_coords))
                    break
                data = line.split(',')
                if flag == 1:
                    x, y = get_coordinate(float(data[3]), float(data[4]), 50)   # 获取横纵坐标
                elif flag == 2:
                    x = float(data[3])
                    y = float(data[4])
                if not last:
                    x_coords.append([x])
                    y_coords.append([y])
                    last = [data[1], data]  # 记录轨迹的起始点
                    continue
                if last[0] != data[1]:
                    x_coords.append([x])
                    y_coords.append([y])
                    last = [data[1], data]
                    if len(x_coords[-1]) > 5:
                        for x, y in zip(x_coords[-2], y_coords[-2]):
                            points[(x, y)] = 1
                    continue

                if abs(int(data[2]) - int(last[1][2])) >= 10:    # abs(x_coords[-1][-1] - x) >= 0.005 or abs(y_coords[-1][-1] - y) >= 0.005
                    x_coords.append([x])
                    y_coords.append([y])
                else:
                    x_coords[-1].append(x)
                    y_coords[-1].append(y)
                last = [data[1], data]

        # 过滤较短的轨迹
        x_coords = [sublist for sublist in x_coords if len(sublist) >= 5]
        y_coords = [sublist for sublist in y_coords if len(sublist) >= 5]

        # with open('D:/Project/traffic/gps_data01-03/road_network_' + time_name + '.jsonl', 'a', encoding='utf-8') as fw:
        #     fw.write(json.dumps({'x': x_coords, 'y':y_coords}, ensure_ascii=False) + '\n')

    if flag == 1:   # 在网格化地图上绘制路网
        plt.figure(figsize=(10, 6))
        for i in tqdm.tqdm(range(len(x_coords))):
            plt.plot(x_coords[i], y_coords[i], marker='o', linestyle='-', linewidth=0.5, markersize=0.1)
        plt.title("Road network")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.savefig('D:/Project/traffic/gps_data01-03/road_network_' + time_name + '.png')
        # plt.show()
    elif flag == 2:
        # 输出边界点
        max_x = 0
        max_y = 0
        min_x = 10000
        min_y = 10000
        for x, y in zip(x_coords, y_coords):
            if max(x) > max_x: max_x = max(x)
            if max(y) > max_y: max_y = max(y)
            if min(y) < min_y: min_y = min(y)
            if min(x) < min_x: min_x = min(x)
        print(max_x, max_y, min_x, min_y)

        trajectories = []
        for x, y in zip(x_coords, y_coords):
            trajectory = []
            for x_point, y_point in zip(x, y):
                trajectory.append((y_point, x_point))
            trajectories.append(trajectory)
        trajectories_new = [[(lat + 0.0024, lon - 0.0025) for lat, lon in trajectory] for trajectory in trajectories]

        m = folium.Map(location=[30.5728, 104.0668], zoom_start=13)
        for i, trajectory in tqdm.tqdm(enumerate(trajectories_new)):
            folium.PolyLine(
                trajectory,
                color='blue',  # 按序分配颜色
                weight=3, opacity=0.8, popup=f'Trajectory {i + 1}'
            ).add_to(m)
        # 保存生成的地图
        # m.save('multi_trajectory_map_.html')


def road_network_plus():
    """
    可过滤低频轨迹
    :return:
    """
    # 加载数据
    time_name = '20161101'
    x_coords = []
    y_coords = []
    if os.path.exists('D:/Project/traffic/gps_data01-03/road_network_' + time_name + '.jsonl'):
        for i in range(1, 4):
            time_name = time_name[:-1] + str(i)
            with open('D:/Project/traffic/gps_data01-03/road_network_' + time_name + '.jsonl', 'r', encoding='utf-8') as f:
                data = json.load(f)
                x_coords += data['x']
                y_coords += data['y']
                print(len(x_coords))

        # 统计频率图
        track = {}
        for x, y in zip(x_coords, y_coords):
            for x_, y_ in zip(x, y):
                track[(x_, y_)] = track.get((x_, y_), 0) + 1

        # 绘图
        plt.figure(figsize=(10, 6))
        for i in tqdm.tqdm(range(len(x_coords))):
            num = 0
            for x, y in zip(x_coords[i], y_coords[i]):
                if track[(x, y)] < 0:
                    num += 1
            if num < 1:
                plt.plot(x_coords[i], y_coords[i], marker='o', linestyle='-', linewidth=0.5, markersize=0.1)
        plt.title("Road network")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.savefig('D:/Project/traffic/gps_data01-03/road_network_' + time_name + '_final.png')
        # plt.show()


def rest_area():
    last_data = []
    x_coords = []
    y_coords = []
    trajectory_idx_list = []
    rest_area = [[[0 for j in range(500)] for i in range(500)] for k in range(4)]       # 4 * 500 * 500
    mode_list = [3600, 7200, 3600, 3600 * 3, 1800, 7200, 1800, 3600 * 3]
    with (open('D:/Project/traffic/gps_data01-03/gps_20161101', 'r') as f):
        # 取数据
        for idx, line in tqdm.tqdm(enumerate(f, 1)):
            data = line.split(',')
            x, y = get_coordinate(float(data[3]), float(data[4]), 50)  # 获取横纵坐标
            data = data + [x, y]
            if not last_data or data[0] != last_data[0]:      # 一个新司机
                if idx % 3000000 <= 1000 < idx:
                    for i_img in range(4):
                        plt.clf()
                        plt.imshow(rest_area[i_img], cmap='viridis', origin='lower', aspect='auto')
                        plt.colorbar()  # 添加颜色条
                        plt.title("Rest area Heatmap")
                        plt.xlim(100, 330)
                        plt.ylim(150, 400)
                        plt.xticks(np.arange(100, 331, 10))
                        plt.yticks(np.arange(150, 401, 10))
                        plt.grid(True)
                        plt.savefig(f'D:/Project/traffic/lsc/data/rest_area/v2/rest_area_{idx}_mode{i_img}.png')
                    break
                if trajectory_idx_list:
                    sorted_trajectory_idx_list = sorted(trajectory_idx_list, key=lambda x: x[0])
                    for last_id, (t, id) in enumerate(sorted_trajectory_idx_list[1:]):
                        for mode_i in range(4):     # 4种时间段模式
                            if mode_list[mode_i * 2] < int(t) - int(sorted_trajectory_idx_list[last_id][0]) < \
                                        mode_list[mode_i * 2 + 1]:
                                # 1~2 / 1~3 / 0.5~2 / 0.5~3
                                x1 = x_coords[sorted_trajectory_idx_list[last_id][1]][-1]   # 前一个订单的下车位置
                                y1 = y_coords[sorted_trajectory_idx_list[last_id][1]][-1]
                                x2 = x_coords[id][0]    # 这个订单的上车位置
                                y2 = y_coords[id][0]
                                center_x = (x1 + x2) / 2.0
                                center_y = (y1 + y2) / 2.0
                                dx = x2 - x1
                                dy = y2 - y1
                                a = math.hypot(dx, dy) / 2.0
                                if a == 0.0:
                                    continue
                                b = a / 2.0
                                theta = math.atan2(dy, dx) if dx != 0 else math.pi / 2
                                cos_theta = math.cos(theta)
                                sin_theta = math.sin(theta)
                                for i in range(500):
                                    for j in range(500):
                                        x_trans = i - center_x
                                        y_trans = j - center_y
                                        x_rot = x_trans * cos_theta + y_trans * sin_theta
                                        y_rot = -x_trans * sin_theta + y_trans * cos_theta
                                        if (x_rot ** 2 / a ** 2 + y_rot ** 2 / b ** 2) <= 1:
                                            rest_area[mode_i][i][j] += 1

                x_coords = [[x]]
                y_coords = [[y]]
                last_data = data
                trajectory_idx_list = [(data[2], 0)]
                continue
            if last_data[1] != data[1]:      # 新的订单
                x_coords.append([x])
                y_coords.append([y])
                last_data = data
                trajectory_idx_list.append((data[2], len(trajectory_idx_list)))
                continue
            x_coords[-1].append(x)
            y_coords[-1].append(y)
            last_data = data
    # plt.imshow(rest_area, cmap='viridis')
    # plt.colorbar()  # 添加颜色条
    # plt.title("500x500 Heatmap")
    # plt.show()


def rest_area_order():
    driver_data = []
    load_flag = False
    # [236,176,89,107]
    rest_area = [[0 for j in range(1000)] for i in range(1000)]

    def rest_area_print(ra):
        real_map = False
        if not real_map:
            min_row = math.inf
            max_row = -math.inf
            min_col = math.inf
            max_col = -math.inf

            for r_idx in range(len(ra)):
                for c_idx in range(len(ra[0])):
                    if ra[r_idx][c_idx] >= 5:
                        min_row = min(min_row, r_idx)
                        max_row = max(max_row, r_idx)
                        min_col = min(min_col, c_idx)
                        max_col = max(max_col, c_idx)

            np_ra = np.array(ra)
            np_ra = np.fliplr(np_ra)
            np_ra = np.flipud(np_ra)
            print(min_row, max_row, min_col, max_col)
            ra = np_ra[min_col: max_col + 1, min_row: max_row + 1].tolist()
            plt.figure(facecolor='white')
            plt.imshow(ra, cmap='viridis', origin='lower', aspect='auto')
            plt.colorbar()
            plt.title("Rest area Heatmap")
            # plt.xticks(np.arange(0, 501, 10))
            # plt.yticks(np.arange(0, 501, 10))
            plt.grid(True)
            # plt.savefig(f'D:/Project/traffic/lsc/data/rest_area/v3/rest_area_order_10000.png')
            plt.show()
        else:
            x0 = 103.94
            y0 = 30.78
            dx = 0.0005223416895        # 50m对应经度增量
            dy = 0.0004509966295        # 50m对应纬度增量
            heat_data = []
            for i in range(1000):
                for j in range(1000):
                    x = x0 + dx * (i - 500)
                    y = y0 + dy * (j - 500)
                    heat_data.append([y, x, rest_area[i][j]])
            m = folium.Map(location=[30.5728, 104.0668], zoom_start=13)
            HeatMap(heat_data, min_opacity=0.1).add_to(m)
            m.save('heatmap_100000.html')

    def driver_rest_area(ra, dd):
        sorted_dd = sorted(dd, key=lambda x: x[2])      # 按起点的时间戳排序
        for idx, d in enumerate(sorted_dd[1:]):
            if 1800 <= int(d[2]) - int(sorted_dd[idx][3]) <= 7200:      # 后订单上车时间 - 前订单下车时间
                x1, y1 = get_coordinate(float(d[4]), float(d[5]), 50)     # 后订单上车位置
                x2, y2 = get_coordinate(float(sorted_dd[idx][6]), float(sorted_dd[idx][7]), 50)     # 前订单下车位置
                x1 = x1 + 500
                x2 = x2 + 500
                y1 = y1 + 500
                y2 = y2 + 500
                if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
                    break
                # print(x1, y1, x2, y2)
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                dx = x2 - x1
                dy = y2 - y1
                a = math.hypot(dx, dy) / 2.0
                if a == 0.0:
                    continue
                b = a / 2.0
                theta = math.atan2(dy, dx) if dx != 0 else math.pi / 2
                cos_theta = math.cos(theta)
                sin_theta = math.sin(theta)
                for i in range(1000):
                    for j in range(1000):
                        x_trans = i - center_x
                        y_trans = j - center_y
                        x_rot = x_trans * cos_theta + y_trans * sin_theta
                        y_rot = -x_trans * sin_theta + y_trans * cos_theta
                        if (x_rot ** 2 / a ** 2 + y_rot ** 2 / b ** 2) <= 1:
                            ra[i][j] += 1

    with open('./data/rest_area/v4/rest_area_data.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data['num'] == 10000:
                rest_area = data['rest_area']
                load_flag = True
                break

    if load_flag:
        rest_area_print(rest_area)
    else:
        with open('D:/Project/traffic/order_data/order_driver_01.txt', 'r', encoding='utf8') as f:
            for idx, line in tqdm.tqdm(enumerate(f, 1)):
                if idx == 10000:
                    with open('./data/rest_area/v4/rest_area_data.jsonl', 'w', encoding='utf8') as fw:
                        fw.write(json.dumps({'num': idx, 'rest_area': rest_area}) + '\n')
                    rest_area_print(rest_area)
                    break
                data = line.split(',')
                if driver_data and driver_data[-1][1] != data[1]:
                    driver_rest_area(rest_area, driver_data)
                    driver_data = []
                driver_data.append(data)


def rest_time():
    last = []
    time_name = '20161101'
    times = []
    rest_time = {}
    trajectory_idx_list = []
    with open('D:/Project/traffic/gps_data01-03/gps_' + time_name, 'r') as f:
        for idx, line in tqdm.tqdm(enumerate(f, 1), desc='Processing lines'):
            data = line.split(',')
            if not last:    # 初始化
                last = data
                trajectory_idx_list.append((data[2], 0))
                times.append(data[2])
                continue

            if last[0] != data[0]:      # 一个新司机
                if idx % 10000000 <= 1000 < idx:
                    break
                times.append(last[2])
                sorted_trajectory_idx_list = sorted(trajectory_idx_list, key=lambda x: x[0])
                max_t = 0
                for last_id, (t, id) in enumerate(sorted_trajectory_idx_list[1:]):
                    tmp_t = int(t) - int(times[sorted_trajectory_idx_list[last_id][1] * 2 + 1])
                    if tmp_t // 60 >= 5: #tmp_t // 60 <= 400 and
                        max_t = max(tmp_t // 3600, max_t)
                if max_t != 0:
                    rest_time[max_t] = rest_time.get(max_t, 0) + 1

                trajectory_idx_list = [(data[2], 0)]
                times = [data[2]]
                last = data
                continue

            if last[1] != data[1]:      # 一个司机的不同订单
                trajectory_idx_list.append((data[2], len(trajectory_idx_list)))
                times.append(last[2])
                times.append(data[2])
                last = data
                continue
            last = data
    x_values = sorted(rest_time.keys())
    y_values = [rest_time[k] for k in x_values]
    plt.bar(x_values, y_values, color='skyblue')

    # 添加标题和轴标签
    plt.title("休息时间统计")
    plt.xlabel("休息时间 分钟")
    plt.ylabel("数量")

    # 显示图表
    plt.show()
    # with open('D:/Project/traffic/gps_data01-03/road_network_' + time_name + '.jsonl', 'a', encoding='utf-8') as fw:
    #     fw.write(json.dumps({'x': x_coords, 'y': y_coords}, ensure_ascii=False) + '\n')


def cal_grid_num(grid):
    min_lat = 1000.0
    max_lat = 0.0
    min_lon = 1000.0
    max_lon = 0.0
    with open('../order_data/order_driver_01.txt', 'r', encoding='utf-8') as f:
        for line in tqdm.tqdm(f):
            data = line.split(',')
            max_lat = max(max_lat, float(data[5]))
            min_lat = min(min_lat, float(data[5]))
            max_lat = max(max_lat, float(data[7]))
            min_lat = min(min_lat, float(data[7]))
            max_lon = max(max_lon, float(data[4]))
            min_lon = min(min_lon, float(data[4]))
            max_lon = max(max_lon, float(data[6]))
            min_lon = min(min_lon, float(data[6]))

    print(min_lat, max_lat, min_lon, max_lon)
    lat_dist = max(get_distance(max_lat, min_lon, min_lat, min_lon),
                   get_distance(max_lat, max_lon, min_lat, max_lon))
    lon_dist = max(get_distance(min_lat, max_lon, min_lat, min_lon),
                   get_distance(max_lat, max_lon, max_lat, min_lon))
    print(lat_dist, lon_dist)
    print(lat_dist // grid, lon_dist // grid)
    """
    29.51723 31.33911 103.25635 104.704311
    201973.00233379198 140378.7875279124
    403.0 280.0
    """


    # def cal_poi():
    df = pd.read_table('../order_data/examine_neighbor_poi_anomaly_detect_chengdu_in_500_1101.txt', header=None)
    poi_type = {}      # 21 类
    grid_type = {}

    for i in range(df.shape[0]):
        poi_str = df.loc[i, 1]
        if isinstance(poi_str, float):
            continue

        poi_list = poi_str.split('|')
        for poi in poi_list:
            if ':' in poi:
                poi_type[poi.split(':')[0]] = 1
            if ';' in poi:
                grid_type[poi.split(';')[1]] = 1

    print(len(poi_type), poi_type)
    print(list(poi_type.keys()))
    print(len(grid_type), grid_type)
    print(list(grid_type.keys()))


def print_order_dist(grid):
    driver_data = {}
    with open('D:/Project/traffic/order_data/order_driver_01.txt', 'r', encoding='utf8') as f:
        for idx, line in tqdm.tqdm(enumerate(f, 1)):
            data = line.split(',')
            data[4], data[5] = get_coordinate(float(data[4]), float(data[5]), grid)
            data[6], data[7] = get_coordinate(float(data[6]), float(data[7]), grid)
            driver_data.setdefault(data[0], []).append(data)
            if len(driver_data) > 300:
                break

    print_data = {}
    for driver_id, order_list in driver_data.items():
        if random.randint(0,9) > 4:
            print_data[driver_id] = []
            for order in order_list:
                print(order[4], order[5], order[6], order[7])
                print_data[driver_id].append((order[4], order[5]))
                print_data[driver_id].append((order[6], order[7]))

    colors = list(mcolors.TABLEAU_COLORS.values())
    if len(print_data) > len(colors):
        random.shuffle(colors)
        colors = colors * (len(print_data) // len(colors) + 1)

    # 创建一个图像和一个子图
    fig, ax = plt.subplots(figsize=(10, 6))
    # 为每个组绘制散点图
    for i, (group_name, points) in enumerate(print_data.items()):
        # 提取 x 和 y 坐标
        x = [point[0] for point in points]
        y = [point[1] for point in points]
        # 绘制当前组的散点图
        ax.scatter(x, y, c=colors[i], label=group_name)

    # 添加标题和标签
    ax.set_title('Scatter Plot of Groups')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')

    # 添加图例
    # ax.legend()
    # 显示网格线
    ax.grid(True, linestyle='--', alpha=0.7)
    # 调整布局
    plt.tight_layout()
    # 显示图像
    plt.show()




if __name__ == '__main__':
    # x0 = 103.94
    # y0 = 30.78
    # dx = 0.0005223416895  # 50m对应经度增量
    # dy = 0.0004509966295  # 50m对应纬度增量
    # print(get_distance(x0, y0, x0, y0 + dy))
    # print(get_distance(x0, y0, x0, y0 - dy))
    # import pdb; pdb.set_trace()

    # 订单分布图
    # print_order_dist(10)

    # poi处理
    # cal_poi(500)

    # 地图范围计算
    # cal_grid_num(500)

    # 休息时间的统计
    # rest_time()

    # 休息区域的统计
    # rest_area()
    rest_area_order()

    # 路网绘制-过滤版
    # road_network_plus()

    # 路网绘制
    # road_network()

    # 单个司机的轨迹
    # single_driver()

