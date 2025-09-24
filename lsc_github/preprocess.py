import pandas as pd
import numpy as np
import folium
import webbrowser
from shapely.geometry import Point, asPolygon
import copy
import math
import matplotlib.pyplot as plt
import json


def get_distance(latitude_a, longitude_a, latitude_b, longitude_b):
    # 给定两个点的经纬度得到两个点之间的距离
    ra = 6378140  # 赤道半径，单位m
    rb = 6356755  # 极半径，单位m
    flatten = (ra - rb) / ra
    # 角度变成弧度
    rad_latitude_a = math.radians(latitude_a)
    rad_longitude_a = math.radians(longitude_a)
    rad_latitude_b = math.radians(latitude_b)
    rad_longitude_b = math.radians(longitude_b)

    try:
        pa = math.atan(rb / ra * math.tan(rad_latitude_a))
        pb = math.atan(rb / ra * math.tan(rad_latitude_b))
        temp = math.sin(pa) * math.sin(pb) + math.cos(pa) * math.cos(pb) * math.cos(rad_longitude_a - rad_longitude_b)
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


class Preprocess:
    driver_point = {}
    driver_time_sequence = {}
    driver_load_factor = {}
    driver_yield_rate = {}
    driver_comfort_matrix = {}
    driver_comfort_matrix_list = {}
    driver_comfort_area_cnt = {}
    general_driver = []
    comfort_driver = []
    str = 'dnkd7bkghbzs1run8g8e2cbcfbzv0jpp'

    # 地图参数
    # 地图左上角点的经纬度，网格划分宽度，矩阵最后宽度
    x0 = 103.94
    y0 = 30.78
    grid_width = 250
    area_width = 25000
    matrix_width = int(area_width / grid_width)

    # 成都网约车价格参数
    # 目前按照1.3元每公里。超过12公里0.64元每公里。时间上0.2元每分钟
    first_level_price_distance = 1.3  # 第一档1.3元每公里
    first_level_boundary = 12  # 第一档截止到12公里
    second_level_price_distance = 0.64  # 第二档0.64元每公里
    time_price = 0.2  # 0.2元每分钟
    min_cost = 8  # 最低消费

    def __init__(self, file_path):
        self.file_path = file_path
        Preprocess.preprocess(self)

    def preprocess(self):
        df = pd.read_table(self.file_path, sep=',')
        # 此处为单文件读取，如果传入文件夹需要额外处理
        df.loc[-1] = df.columns
        df.index = df.index + 1
        df.sort_index(inplace=True)
        df.columns = ['orderID', 'driverID', 'beginTime', 'endTime', 'beginLongitude', 'beginLatitude', 'endLongitude',
                      'endLatitude']
        Preprocess.divide_by_driver(self, df)
        print("Divide_by_driver has finished! We have %d drivers!" % (len(self.driver_point)))
        Preprocess.get_comfort_boundary(self)
        print("Get_comfort_boundary has finished!")
        Preprocess.get_load_factor_and_yield_rate(self)
        print("Get_load_factor_and_yield_rate has finished!")
        Preprocess.driver_divide(self)
        print("Driver_divide has finished!")
        Preprocess.save_data(self)
        print("Save_data has finished!")

    def divide_by_driver(self, df):
        for order in df.values:
            # driver_point保存起止点，driver_time_sequence保存起止时间
            self.driver_point.setdefault(order[1], []).append([float(order[4]), float(order[5])])
            self.driver_point.setdefault(order[1], []).append([float(order[6]), float(order[7])])
            self.driver_time_sequence.setdefault(order[1], []).append(float(order[2]))
            self.driver_time_sequence.setdefault(order[1], []).append(float(order[3]))

    def get_comfort_boundary(self):
        # 先按照一个司机的所有点来构造一个奇怪的多边形，之后根据这个多边形找到极点，利用极点重新构造一个正常的多边形
        # 如果该司机只有一个或者两个订单，那么构不成多边形或者效果很差，直接舍弃数据
        # 如果该司机有三个及以上订单，那么算出舒适矩阵，再根据舒适矩阵去判断司机类型
        cnt = 0
        for each in self.driver_point.keys():
            cnt += 1
            # 因为这一步非常慢因此适当输出进度
            if cnt % 1000 == 0:
                print(cnt)

            # print(each + str(cnt))
            points = np.array(copy.deepcopy(self.driver_point[each]))
            points_num = len(points)
            if points_num <= 4:
                continue
            boundary = asPolygon(points).convex_hull  # 最小凸包的Polygon表示，在后面判断网格的时候可以用这个
            Preprocess.get_comfort_matrix(self, each, boundary)

            '''if each == self.str:  # 可视化str对应的舒适圈以及研究范围
                boundary_list = list(boundary.exterior.coords)
                points_list = list(points)
                m = folium.Map(location=[points_list[0][1], points_list[0][0]], zoom_start=11)
                for i in range(0, points_num):
                    folium.Marker(location=[points_list[i][1], points_list[i][0]],
                                  icon=folium.Icon(color='blue')).add_to(m)
                for i in range(0, len(boundary_list) - 1):
                    folium.Marker(location=[boundary_list[i][1], boundary_list[i][0]], icon=folium.Icon(color='red')). \
                        add_to(m)
                    folium.PolyLine(locations=[[boundary_list[i][1], boundary_list[i][0]],
                                               [boundary_list[i + 1][1], boundary_list[i + 1][0]]], color='red').add_to(
                        m)
                folium.Polygon(locations=[[30.78, 103.94], [30.56, 103.94], [30.56, 104.20],
                                          [30.78, 104.20]], color='green').add_to(m)
                print('###')
                print(boundary_list)
                m.save('temp.html')
                webbrowser.open('temp.html')
                plt.matshow(self.driver_comfort_matrix[self.str])
                plt.savefig('comfort_matrix.jpg')
                plt.show()'''

    def get_comfort_matrix(self, driver, boundary):
        # 这部分是将已经确定的多边形舒适圈网格化，用01矩阵来表示其舒适圈
        # 四个角的范围点分别是：
        # [30.78, 103.94]（左上）
        # [30.56, 104.20]（右下）
        # 按照左上角为（0,0）的矩阵，最左边经度最小，最上面纬度最大，所以左上角的点应该是第一个点
        # 经度为i(x)，纬度为j(y)。按照25000m*25000m。暂定250m为网格跨度，那么就是100*100
        boundary_list = list(boundary.exterior.coords)
        # 将边界点的经纬度变成边界点的网格坐标
        boundary_mesher = []
        for i in range(0, len(boundary_list) - 1):
            # print(boundary_list[i])
            # print(get_distance(y0, x0, boundary_list[i][1], boundary_list[i][0]))
            # distance_x = distance_y = 0
            # 计算经度差，但是经度差对应的是矩阵的纵坐标。所以x其实对应的是纵坐标。
            # 0,0 0,1 0,2 ... 0,99
            # 1,0 1,1 1,2 ... 1,99
            # ...
            # 99,0 99,1 99,2 ... 99,99

            '''# 这种判断方式是将边界点直接拉进网格内，如果一个点的坐标是（-1,3），那么直接拉进来变成（0,3）。不推荐这种判断方式，
            下面那种比较好。
            if boundary_list[i][0] < x0:
                distance_x = -1
                x = 0
            else:
                distance_x = get_distance(y0, x0, y0, boundary_list[i][0])
                x = int(distance_x / grid_width)
                if x >= matrix_width:
                    x = matrix_width - 1
            # 计算纬度差。所以y其实对应的是横坐标
            if boundary_list[i][1] > y0:
                distance_y = -1
                y = 0
            else:
                distance_y = get_distance(y0, x0, boundary_list[i][1], x0)
                y = int(distance_y / grid_width)
                if y >= matrix_width:
                    y = matrix_width - 1'''

            # 这种判断方式保留了原来的点，允许他们是负坐标，只是统计网格数和舒适区的时候只考虑我们规定的范围内的
            distance_x = get_distance(self.y0, self.x0, self.y0, boundary_list[i][0])
            if distance_x == -1:
                continue
            x = int(distance_x / self.grid_width)
            if boundary_list[i][0] < self.x0:
                x *= -1
            distance_y = get_distance(self.y0, self.x0, boundary_list[i][1], self.x0)
            if distance_y == -1:
                continue
            y = int(distance_y / self.grid_width)
            if boundary_list[i][1] > self.y0:
                y *= -1
            # 添加的时候注意y是横坐标，x是纵坐标
            boundary_mesher.append((y, x))
            # print(distance_x, distance_y, x, y)
        # print("-------")
        # print(boundary)
        if len(boundary_mesher) < 3:
            return
        polygon = asPolygon(boundary_mesher)
        # print(polygon)
        grid_cnt = 0
        matrix = np.zeros((self.matrix_width, self.matrix_width))
        '''for i in range(0, self.matrix_width):
            for j in range(0, self.matrix_width):
                # if polygon.contains(Point(i, j)):  # 不算边界点
                if not polygon.disjoint(Point(i, j)):  # 算上边界点
                    # 舒适矩阵
                    grid_cnt += 1
                    matrix[i][j] = 1'''
        # 适当优化了下。因为凸多边形对于每一行只要找到最左边点和最右边点，中间的就都是多边形内部的点
        for i in range(0, self.matrix_width):
            begin = -1
            end = -1
            for j in range(0, self.matrix_width):
                if not polygon.disjoint(Point(i, j)):
                    begin = j
                    break
            if begin == -1:  # 这行没多边形
                continue
            for j in range(self.matrix_width - 1, -1, -1):
                if not polygon.disjoint(Point(i, j)):
                    end = j
                    break
            if end == -1:  # 讲道理这两种情况应该不会出现，但还是写上。
                continue
            if begin > end:
                continue

            for j in range(begin, end + 1):
                # 舒适矩阵
                grid_cnt += 1
                matrix[i][j] = 1

        self.driver_comfort_area_cnt[driver] = grid_cnt
        self.driver_comfort_matrix[driver] = matrix
        self.driver_comfort_matrix_list[driver] = matrix.tolist()

    def get_load_factor_and_yield_rate(self):
        # 求司机的载客率。对于每个订单，终止时间必然大于起始时间，而下一单的起始时间必然大于这一单的终止时间。因此，虽然订单并不是按照
        # 时间顺序排序的，但是只要把所有的时间戳按照升序排序，第2i个时间就是起始时间，第2i+1个时间就是终止时间。不过这样计算载客率，
        # 不考虑第一单之前和最后一单之后空载时间
        for each in self.driver_time_sequence.keys():
            time_copy = copy.deepcopy(self.driver_time_sequence[each])
            time_copy.sort()
            load_time = 0
            empty_time = 0
            yield_time = 0
            loop_cnt = int(len(time_copy) / 2) - 1
            # 一共有int(len(time_copy) / 2)个订单，但是先不考虑最后一个订单，因为最后一个订单只能用来计算载客率
            # 载客时间算loop_cnt + 1次，空载时间算loop_cnt次
            # 这里需要计算时间对应的费用
            for i in range(0, loop_cnt):
                load_time += time_copy[2 * i + 1] - time_copy[2 * i]
                empty_time += time_copy[2 * i + 2] - time_copy[2 * i + 1]
                yield_time += ((time_copy[2 * i + 1] - time_copy[2 * i]) / 60) * self.time_price
            load_time += time_copy[2 * loop_cnt + 1] - time_copy[2 * loop_cnt]
            self.driver_load_factor[each] = load_time / (load_time + empty_time)

            # 下面对这个司机的载客点卸客点进行统计。需要用到上述的load_time和empty_time作为运营时间
            # 这里只需要计算里程对应的费用
            driver_point_copy = copy.deepcopy(self.driver_point[each])
            service_time = load_time + empty_time
            yield_distance = 0
            for i in range(0, int(len(driver_point_copy) / 2)):
                distance_1 = get_distance(driver_point_copy[2*i][0], driver_point_copy[2*i][1],
                                          driver_point_copy[2*i+1][0], driver_point_copy[2*i][1])
                distance_2 = get_distance(driver_point_copy[2*i+1][0], driver_point_copy[2*i][1],
                                          driver_point_copy[2*i+1][0], driver_point_copy[2*i+1][1])
                distance = distance_1 + distance_2
                distance /= 1000  # 转成公里计算
                if distance < self.first_level_boundary:
                    yield_distance += distance * self.first_level_price_distance
                else:
                    yield_distance += self.first_level_boundary * self.first_level_price_distance + \
                                      (distance - self.first_level_boundary) * self.second_level_price_distance
            yield_sum = yield_distance + yield_time
            if yield_sum < self.min_cost:  # 最低消费
                yield_sum = self.min_cost
            yield_rate = yield_sum / (service_time / 3600)  # 司机每小时能挣多少钱
            self.driver_yield_rate[each] = yield_rate

    def driver_divide(self):
        # 判断舒适矩阵包括网格数为x的司机有多少人，以此为依据进行统计
        # 选取一个合适的阈值区分舒适司机和非舒适司机，这里先把所有的都认为是舒适司机
        divide_width = 10
        area_cnt_statistic = np.zeros(int(self.matrix_width * self.matrix_width / divide_width) + 1)
        for each in self.driver_comfort_area_cnt:
            area_cnt_statistic[int(self.driver_comfort_area_cnt[each] / divide_width)] += 1
            self.comfort_driver.append(each)
        filename = 'area_cnt_statistic.json'
        with open(filename, 'w') as file_obj:
            json.dump(area_cnt_statistic.tolist(), file_obj)
        plt.bar(range(len(area_cnt_statistic)), area_cnt_statistic)
        plt.savefig('cnt_statistic.jpg')
        # plt.show()

    def save_data(self):
        # 存储文件
        filename = 'driver_point.json'
        with open(filename, 'w') as file_obj:
            json.dump(self.driver_point, file_obj, indent=4)

        filename = 'driver_time_sequence.json'
        with open(filename, 'w') as file_obj:
            json.dump(self.driver_time_sequence, file_obj, indent=4)

        filename = 'driver_load_factor.json'
        with open(filename, 'w') as file_obj:
            json.dump(self.driver_load_factor, file_obj, indent=4)

        filename = 'driver_yield_rate.json'
        with open(filename, 'w') as file_obj:
            json.dump(self.driver_yield_rate, file_obj, indent=4)

        filename = 'driver_comfort_area_cnt.json'
        with open(filename, 'w') as file_obj:
            json.dump(self.driver_comfort_area_cnt, file_obj, indent=4)

        filename = 'general_driver.json'
        with open(filename, 'w') as file_obj:
            json.dump(self.general_driver, file_obj, indent=4)

        filename = 'comfort_driver.json'
        with open(filename, 'w') as file_obj:
            json.dump(self.comfort_driver, file_obj, indent=4)

        filename = 'driver_comfort_matrix_list.json'
        with open(filename, 'w') as file_obj:
            json.dump(self.driver_comfort_matrix_list, file_obj, indent=4)
