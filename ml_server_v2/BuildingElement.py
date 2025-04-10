import sys
sys.path.append("..")
from ml_server_v2.BuildingSpaceBase import *
from ml_server_v2.SpaceStandard import SpaceStandard
from apocolib.MlLogger import mlLogger as ml_logger
from apocolib.CouchDBPool import *
from apocolib.dataset_io import save_dataset, load_dataset, append_to_dataset
import numpy as np
import time
import random
import json
import requests
import json
import os
import copy
import tensorflow as tf
import pdb
from abc import ABC, abstractmethod
import string


_ss = SpaceStandard()  # 创建空间标准对象

# 建筑基本空间基类，墙，房间，楼层，建筑


class BuildingElement:
    def __init__(self, area=0.0, height=0.0, orientation=None):
        self._area = round(area, 4)
        self._height = round(height, 4)
        self._orientation = orientation

    def get_area(self):
        return self._area

    def get_height(self):
        return self._height

    def get_orientation(self):
        return self._orientation

    def set_orientation(self, orientation):
        self._orientation = orientation

    @abstractmethod
    def to_json(self):
        pass

    @abstractmethod
    def to_json_cn(self):
        pass

    @abstractmethod
    def to_tensor(self):
        pass

    def save_to_json(self, file_path, json_method='to_json'):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if json_method == 'to_json':
            json_data = self.to_json()
        elif json_method == 'to_json_cn':
            json_data = self.to_json_cn()
        else:
            raise ValueError(f"Invalid value for json_method: {json_method}")

        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(json_data, json_file, ensure_ascii=False,
                      indent=4, cls=self.CustomJSONEncoder)

    class CustomJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, set):
                return list(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            return super().default(obj)


class Floor(BuildingElement):

    HOUSE_NUMBER = 12  # 最多12个房间
    STAIRCASE_NUMBER = 4  # 最多2个楼梯间
    CORRIDORS_NUMBER = 4  # 最多2个公共部位

    FEATURES_LEN = 27360

    def __init__(self, name=None, floor_id=0, area=0, out_wall_area=0, house_num=0, houses=[],
                 staircase_num=1, staircases=[], corridor_num=0, corridors=[],height=2.95):
        self._name = name
        self._floor_id = floor_id
        self._area = area
        self._out_wall_area = out_wall_area
        self._house_num = house_num
        self._staircase_num = staircase_num
        self._corridor_num = corridor_num
        self._houses = houses
        self._staircases = staircases
        self._corridors = corridors

        self.calculated = False
        self._height = height

    def get_name(self):
        return self._name

    def get_flood_id(self):
        return self._floor_id

    def get_house_num(self):
        self._house_num = len(self._houses)
        return self._house_num

    def get_staircase_num(self):
        return self._staircase_num

    def get_corridor_num(self):
        return self._corridor_num

    def get_houses(self):
        return self._houses

    def get_staircases(self):
        return self._staircases

    def get_corridors(self):
        return self._corridors

    def add_house(self, house):
        self._houses.append(house)

    def add_staircase(self, staircase):
        # 检查 staircase 是否为 publicSpace 类型
        if not isinstance(staircase, PublicSpace):
            raise ValueError(f"Invalid value for staircase: {staircase}")
        self._staircases.append(staircase)

    def add_corridor(self, corridor):
        # 检查 staircase 是否为 publicSpace 类型
        if not isinstance(corridor, PublicSpace):
            raise ValueError(f"Invalid value for corridor: {corridor}")
        self._corridors.append(corridor)

    def set_house(self, houses):
        self._houses = houses

    def update_house(self, house, index):
        self._houses[index] = house

    def set_staircase(self, staircases):
        self._staircases = staircases

    def update_staircase(self, staircase, index):
        self._staircases[index] = staircase

    def set_corridor(self, corridors):
        self._corridors = corridors

    def update_corridor(self, corridor, index):
        self._corridors[index] = corridor

    def calculate_all(self):

        self._area = 0
        self._house_num = 0
        self._staircase_num = 0
        self._corridor_num = 0
        self._total_out_wall_area = 0
        self._total_window_area = 0
        self._total_cost = 0
        self._avg_ww_ratio = 0
        self._total_avg_k = 0

        t_k = 0

        for house in self._houses:

            if isinstance(house, House):
                house.calculate_all()
                self._house_num += 1
                self._area += house.get_area()
                self._total_out_wall_area += house.get_total_wall_area()
                self._total_window_area += house.get_total_window_area()
                self._total_cost += house.get_total_cost()

                t_k += house.get_avg_k()

        for staircase in self._staircases:

            if isinstance(staircase, PublicSpace):
                #staircase.caculate_all()
                self._staircase_num += 1
                self._area += staircase.get_area()
                self._total_out_wall_area += staircase.get_total_wall_area()
                self._total_window_area += staircase.get_total_window_area()
                self._total_cost += staircase.get_total_cost()

                t_k += staircase.get_avg_k()


               
        for corridor in self._corridors:

            if isinstance(corridor, PublicSpace):
                #corridor.caculate_all()
                self._corridor_num += 1
                self._area += corridor.get_area()
                self._total_out_wall_area += corridor.get_total_wall_area()
                self._total_window_area += corridor.get_total_window_area()
                self._total_cost += corridor.get_total_cost()

                t_k += corridor.get_avg_k()

        self._avg_ww_ratio = self._total_window_area / self._total_out_wall_area

        self._total_avg_k = t_k / self._area

        self.calculated = True

    # 总外墙面积，计算所有houses的外墙面积之和
    def get_out_wall_area(self):

        self._out_wall_area = 0
        for house in self._houses:
            if isinstance(house, House):
                self._out_wall_area += house.get_total_wall_area()

        for staircase in self._staircases:

            #if isinstance(staircase, PublicSpace):
            self._out_wall_area += staircase.get_total_wall_area()
        
        for corridor in self._corridors:
            #if isinstance(corridor, PublicSpace):
            self._out_wall_area += corridor.get_total_wall_area()

        self._out_wall_area = round(self._out_wall_area, 4)

        return self._out_wall_area

    def get_total_area(self):

        self._area = 0
        for house in self._houses:
            #ml_logger.info(f"house.get_area {house.get_area}")
            self._area += house.get_total_area()

        for staircase in self._staircases:
            self._area += staircase.get_area()

        for corridor in self._corridors:
            self._area += corridor.get_area()

        self._area = round(self._area, 4)

        return self._area

    def get_total_window_area(self):

        self._total_window_area = 0
        for house in self._houses:
            self._total_window_area += house.get_total_window_area()

        for staircase in self._staircases:
            self._total_window_area += staircase.get_total_window_area()

        for corridor in self._corridors:
            self._total_window_area += corridor.get_total_window_area()

        self._total_window_area = round(self._total_window_area, 4)
        return self._total_window_area

    def get_total_cost(self):
        self._total_cost = 0
        for house in self._houses:
            self._total_cost += house.get_total_cost()

        for staircase in self._staircases:
            self._total_cost += staircase.get_total_cost()

        for corridor in self._corridors:
            self._total_cost += corridor.get_total_cost()

        self._total_cost = round(self._total_cost, 4)
        return self._total_cost

    def get_avg_ww_ratio(self):
        self._avg_ww_ratio = 0
        for house in self._houses:
            self._avg_ww_ratio += house.get_avg_ww_ratio() * house.get_area()

        for staircase in self._staircases:
            self._avg_ww_ratio += staircase.get_avg_ww_ratio() * staircase.get_area()

        for corridor in self._corridors:
            self._avg_ww_ratio += corridor.get_avg_ww_ratio() * corridor.get_area()

        if self.get_total_area() == 0:
            return 0

        self._avg_ww_ratio = self._avg_ww_ratio / self.get_total_area()
        self._avg_ww_ratio = round(self._avg_ww_ratio, 4)
        return self._avg_ww_ratio

    def get_avg_k(self):
        self._avg_k = 0
        for house in self._houses:
            self._avg_k += house.get_avg_k() * house.get_area()

        for staircase in self._staircases:
            self._avg_k += staircase.get_avg_k() * staircase.get_area()

        for corridor in self._corridors:
            self._avg_k += corridor.get_avg_k() * corridor.get_area()

        if self.get_total_area() == 0:
            return 0

        self._avg_k = self._avg_k / self.get_total_area()

        return self._avg_k

    def get_cost_view(self):
        # 汇总所有房间的cost_view 中相同的key 的 value 值 累加
        #         key = str(material_type)
        # self._cost_view[key] = {
        #     'area': round(material.get_area(), 4),
        #     'price': material.get_price(),
        #     'cost': round(material.get_cost(), 2)
        # }
        # 根据上述结构，将所有房间的cost_view 汇总

        # pdb.set_trace()

        self._cost_view = {}
        for house in self._houses:
            for key in house.get_cost_view():

                assert key is not None and key != '', f"key is None or ''"

                # print("house key ", key)
                if key is None or key == '':
                    continue

                if key in self._cost_view:
                    self._cost_view[key]['area'] += house.get_cost_view()[key]['area']
                    self._cost_view[key]['cost'] += house.get_cost_view()[key]['cost']
                else:
                    # self._cost_view.append(key)
                    self._cost_view[key] = {
                        'area': round(house.get_cost_view()[key]['area'], 4),
                        'price': round(house.get_cost_view()[key]['price'], 2),
                        'cost': round(house.get_cost_view()[key]['cost'], 4)
                    }

        for staircase in self._staircases:
            for key in staircase.get_cost_view():
                assert key is not None and key != '', f"key is None or ''"

                # print("staircase key ", key)
                if key is None or key == '':
                    continue

                if key in self._cost_view:
                    self._cost_view[key]['area'] += staircase.get_cost_view()[key]['area']
                    self._cost_view[key]['cost'] += staircase.get_cost_view()[key]['cost']
                else:
                    # self._cost_view.append(key)
                    self._cost_view[key] = {
                        'area': round(staircase.get_cost_view()[key]['area'], 4),
                        'price': round(staircase.get_cost_view()[key]['price'], 2),
                        'cost': round(staircase.get_cost_view()[key]['cost'], 4)
                    }

        for corridor in self._corridors:
            for key in corridor.get_cost_view():
                assert key is not None and key != '', f"key is None or ''"

                # print("corridor key ", key)
                if key is None or key == '':
                    continue

                if key in self._cost_view:
                    self._cost_view[key]['area'] += corridor.get_cost_view()[key]['area']
                    self._cost_view[key]['cost'] += corridor.get_cost_view()[key]['cost']
                else:
                    # self._cost_view.append(key)
                    self._cost_view[key] = {
                        'area': round(corridor.get_cost_view()[key]['area'], 4),
                        'price': round(corridor.get_cost_view()[key]['price'], 2),
                        'cost': round(corridor.get_cost_view()[key]['cost'], 4)
                    }

        return self._cost_view

    def __str__(self):
        return f"Floor name:{self._name}, house num:{self._house_num}, houses:{self._houses}"

    def __repr__(self):  # 用于生成对象的字符串表示
        return self.__str__()

    def to_tensor(self):

        _x = []
        _y = []

        # 生成一个大的tensor，每个house是一个小的tensor

        #ml_logger.info(f"convert house num: {len(self._houses)} to tensor ....")
        
        x_len = 0
        y_len = 0
        for house in self._houses:
            houses_features, target = house.to_tensor()
            
            x_len = len(houses_features)
            y_len = len(target)
            
            houses_features = tf.cast(houses_features, dtype=tf.float64)
            target = tf.cast(target, dtype=tf.float64)
            _x.append(houses_features)
            _y.append(target)

        # 补满空空间
        #ml_logger.info(f"convert empty house num: {Floor.HOUSE_NUMBER - len(self._houses)} to tensor ....")
        #for i in range(Floor.HOUSE_NUMBER - len(self._houses)):
            #empty_tensors, empty_targets = House.get_empty_house_tensor()
            # 直接按照长度补0
        empty_tensors = tf.cast([0] * x_len * (Floor.HOUSE_NUMBER - len(self._houses)), dtype=tf.float64)
        empty_targets = tf.cast([0] * y_len * (Floor.HOUSE_NUMBER - len(self._houses)), dtype=tf.float64)

        _x.append(empty_tensors)
        _y.append(empty_targets)

        # staircases 的tensor，2023.11.1
        #ml_logger.info(f"convert staircase num: {len(self._staircases)} to tensor ....")
        for staircase in self._staircases:
            staircase_features, target = staircase.to_tensor()
            staircase_features = tf.cast(staircase_features, dtype=tf.float64)
            target = tf.cast(target, dtype=tf.float64)
            _x.append(staircase_features)
            _y.append(target)

        #ml_logger.info(f"convert empty staircase num: {Floor.STAIRCASE_NUMBER - len(self._staircases)} to tensor ....")
        # 补满staircases空空间 2023.11.1，最多 4个
        #for i in range(Floor.STAIRCASE_NUMBER - len(self._staircases)):
        #    empty_tensors, empty_targets = PublicSpace.get_empty_room_tensor()
        empty_tensors = tf.cast([0] * PublicSpace.ROOM_FEATURES_LEN * (Floor.STAIRCASE_NUMBER - len(self._staircases)), dtype=tf.float64)
        empty_targets = tf.cast([0] * PublicSpace.ROOM_TARGETS_LEN * (Floor.STAIRCASE_NUMBER - len(self._staircases)), dtype=tf.float64)

        _x.append(empty_tensors)
        _y.append(empty_targets)

        #ml_logger.info(f"convert corridor num: {len(self._corridors)} to tensor ....")
        # 增加corridors 的tensor，2023.11.1
        for corridor in self._corridors:
            corridor_features, target = corridor.to_tensor()
            corridor_features = tf.cast(corridor_features, dtype=tf.float64)
            target = tf.cast(target, dtype=tf.float64)
            _x.append(corridor_features)
            _y.append(target)

        #ml_logger.info(f"convert empty corridor num: {Floor.CORRIDORS_NUMBER - len(self._corridors)} to tensor ....")
        # 补满corridors空空间 2023.11.1，最多 4个
        #for i in range(Floor.CORRIDORS_NUMBER - len(self._corridors)):
        #    empty_tensors, empty_targets = PublicSpace.get_empty_room_tensor()

        empty_tensors = tf.cast([0] * PublicSpace.ROOM_FEATURES_LEN * (Floor.CORRIDORS_NUMBER - len(self._corridors)), dtype=tf.float64)
        empty_targets = tf.cast([0] * PublicSpace.ROOM_TARGETS_LEN * (Floor.CORRIDORS_NUMBER - len(self._corridors)), dtype=tf.float64)

        _x.append(empty_tensors)
        _y.append(empty_targets)

        #ml_logger.info("add avg_cost and avg_k to tensor ....")        
        # 优化为单位面积平均成本，cost/m2 ，sunhy 2023.10.24
        _y.append([round(float(self.get_total_cost()/self.get_total_area()),
                  4), round(float(self.get_avg_k()), 4)])

        _x = tf.concat(_x, axis=0)
        _y = tf.concat(_y, axis=0)
        _y = tf.cast(_y, tf.float64)

        #ml_logger.info(
        #    f"floor features shape = {_x.shape} ,targets shape = {_y.shape} ")
        return _x, _y

    #
    def tensor_to_floor(self, floor_tensors):

        #pdb.set_trace()

        # print(f"floor_tensors shape = {floor_tensors.shape}")
        # print(f"floor_tensors = {floor_tensors}")
        #_x = []
        #_y = []
        house_tensors, staircase_tensors, corridor_tensors = extract_building_features(floor_tensors)


        for house_index, house in enumerate(self._houses):

            if isinstance(house, House):

                h_features = house_tensors[house_index*House.HOUSE_FEATURES:(house_index + 1) * House.HOUSE_FEATURES]
                #house = house.tensor_to_house(h_features, house_index)
                if house.get_total_area() == 0:
                    ml_logger.warning(f"house {house.get_name()} area is 0, skip it")
                    continue

                house = house.tensor_to_house(h_features)
                self.update_house(copy.deepcopy(house), house_index)

        for staircase_index, staircase in enumerate(self._staircases):

            if isinstance(staircase, PublicSpace):

                if staircase.get_area() == 0:
                    ml_logger.warning(f"staircase {staircase.get_name()} area is 0, skip it")
                    continue
                
                s_features = staircase_tensors[staircase_index *PublicSpace.ROOM_FEATURES_LEN:
                                                (staircase_index + 1) * PublicSpace.ROOM_FEATURES_LEN]
                staircase = staircase.tensor_to_room(s_features)
                self.update_staircase(copy.deepcopy(staircase), staircase_index)


        for corridor_index, corridor in enumerate(self._corridors):

            if isinstance(corridor, PublicSpace):

                if corridor.get_area() == 0:
                    ml_logger.warning(f"corridor {corridor.get_name()} area is 0, skip it")
                    continue

                c_features = corridor_tensors[corridor_index*PublicSpace.ROOM_FEATURES_LEN:
                                                (corridor_index + 1) * PublicSpace.ROOM_FEATURES_LEN]
                corridor = corridor.tensor_to_room(c_features)
                self.update_corridor(copy.deepcopy(corridor), corridor_index)


        self.calculate_all()

        #self._total_cost = floor_tensors[-2]
        #self._avg_k = floor_tensors[-1]
        return self

    def to_json(self):

        return {
            "floor_name": self._name,
            "floor_id": self._floor_id,
            "house_num": self.get_house_num(),
            "floor_area": self.get_total_area(),  # 总面积
            "floor_out_wall_area": self.get_out_wall_area(),  # 总外墙面积
            "total_window_area": self.get_total_window_area(),  # 总窗面积
            # "total_glass_area": self.get_total_glass_area(),  # 总玻璃面积
            "total_cost": round(float(self.get_total_cost()), 4),  # 总造价
            "avg_ww_ratio": round(float(self.get_avg_ww_ratio()), 4),  # 平均窗墙比
            "avg_k": round(float(self.get_avg_k()), 4),  # 平均K值
            "houses": [house.to_json() for house in self._houses],
            # 楼梯间
            "staircases": [staircase.to_json() for staircase in self._staircases],
            # 走廊
            "corridors": [corridor.to_json() for corridor in self._corridors]

        }

    def to_json_cn(self):
        return {
            "floor_name(楼层名称)": self._name,
            "floor_id(楼层id)": self._floor_id,
            "house_num(房屋数量)": self.get_house_num(),
            "floor_area(楼层面积)": self.get_total_area(),  # 总面积
            "floor_out_wall_area(外墙面积)": self.get_out_wall_area(),  # 总外墙面积
            "total_window_area(总窗面积)": self.get_total_window_area(),  # 总窗面积
            # "total_glass_area(总玻璃面积)": self.get_total_glass_area(),  # 总玻璃面积
            "total_cost(总造价)": round(float(self.get_total_cost()), 4),  # 总造价
            # 平均窗墙比
            "avg_ww_ratio(平均窗墙比)": round(float(self.get_avg_ww_ratio()), 4),
            "avg_k(平均K值)": round(float(self.get_avg_k()), 4),  # 平均K值
            "houses(房屋)": [house.to_json_cn() for house in self._houses],
            # 楼梯间
            "staircases(楼梯间)": [staircase.to_json_cn() for staircase in self._staircases],
            # 走廊
            "corridors(走廊)": [corridor.to_json_cn() for corridor in self._corridors]
        }

    def json_to_floor(self, json_data):  # 根据json 生产floor 对象
        self._name = json_data["floor_name"]
        self._floor_id = json_data["floor_id"]
        self._houses = []
        self._area = json_data["floor_area"]
        self._out_wall_area = json_data["floor_out_wall_area"]
        self._total_window_area = json_data["total_window_area"]
        self._house_num = json_data["house_num"]

        self._staircases = []
        self._corridors = []

        for house in json_data["houses"]:
            self._houses.append(House().json_to_house(house))

        for staircase in json_data["staircases"]:
            self._staircases.append(Room().json_to_room(staircase))

        for corridor in json_data["corridors"]:
            self._corridors.append(Room().json_to_room(corridor))

        json_out_wall_area = json_data["floor_out_wall_area"]

        return self


class Building(BuildingElement):
    def __init__(self, id=0, name='', area=0, out_wall_area=0, floor_num=1, house_num=1,
                 staircase_num=1, corridor_num=0):
        self._id = id
        self._name = name
        self._area = area
        self._out_wall_area = out_wall_area
        self._floor_num = floor_num
        self._house_num = house_num
        self._staircase_num = staircase_num
        self._corridor_num = corridor_num

        self._floors = []
        # yyyy-mm-dd hh:mm:ss
        self.create_time = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime())

        self.couchdb_doc_id = None

        self._form_factor = 0 #  S=F0/V0式中:S—建筑体型系数F0—建筑的外表面积V0—建筑体积


    def get_form_factor(self):
        
        if self._floors is None or len(self._floors) == 0:
            ml_logger.error("self._floors is none")
            return 0
        b_v = self._floors[0].get_total_area()*self._floor_num*self._floors[0].get_height()

        if b_v == 0:
            ml_logger.error("b_v is 0")
            return 0

        self._form_factor = round((self.get_out_wall_area() + self._floors[0].get_total_area()) / b_v, 4) # 简单计算，周围外表面积+楼顶面积（按层面积简单计算）

        #print("self._form_factor", self._form_factor, self.get_out_wall_area(), self._floors[0].get_total_area(), b_v)
        return self._form_factor

    def get_name(self):
        return self._name

    def get_id(self):
        return self._id

    def set_doc_id(self, doc_id):
        self.couchdb_doc_id = doc_id

    def get_doc_id(self):
        return self.couchdb_doc_id

    def get_floor_num(self):
        return self._floor_num

    def get_house_num(self):
        self._house_num = 0
        for floor in self._floors:
            self._house_num += floor.get_house_num()
        return self._house_num

    def get_floors(self):
        return self._floors

    def add_floor(self, floor):
        self._floors.append(floor)

    def update_floor(self,floor, index):
        self._floors[index] = floor

    def get_total_area(self):
        self._total_area = 0
        for floor in self._floors:
            self._total_area += floor.get_total_area()
            #print(f"floor.get_total_area() {floor.get_total_area()}")
        self._total_area = round(self._total_area, 4)
        return self._total_area

    def get_out_wall_area(self):
        self._out_wall_area = 0
        for floor in self._floors:
            self._out_wall_area += floor.get_out_wall_area()
        self._out_wall_area = round(self._out_wall_area, 4)
        return self._out_wall_area

    def get_total_window_area(self):
        self._total_window_area = 0
        for floor in self._floors:
            self._total_window_area += floor.get_total_window_area()
        self._total_window_area = round(self._total_window_area, 4)
        return self._total_window_area

    def get_total_cost(self):
        self._total_cost = 0
        for floor in self._floors:
            self._total_cost += floor.get_total_cost()
        self._total_cost = round(self._total_cost, 4)
        return self._total_cost

    # 所有窗户面积之和与所有外墙面积之和的比值
    def get_avg_ww_ratio(self):
        self._avg_ww_ratio = 0.0

        if self._out_wall_area == 0:
            self._avg_ww_ratio = 0
            return self._avg_ww_ratio

        self._avg_ww_ratio = round(self._total_window_area/self._out_wall_area,2)

        return self._avg_ww_ratio

        # for floor in self._floors:
        #     self._avg_ww_ratio += floor.get_avg_ww_ratio()*floor.get_out_wall_area()

        # if self.get_out_wall_area() == 0:
        #     self._avg_ww_ratio = 0
        #     return self._avg_ww_ratio

        # self._avg_ww_ratio = round(
        #     self._avg_ww_ratio/self.get_out_wall_area(), 4)
        # return self._avg_ww_ratio

        # self._avg_ww_ratio = 0
        # for floor in self._floors:
        #     self._avg_ww_ratio += floor.get_avg_ww_ratio()
        # self._avg_ww_ratio = round(self._avg_ww_ratio, 4)
        # return self._avg_ww_ratio

    def get_avg_k(self):
        self._avg_k = 0
        for floor in self._floors:
            self._avg_k += floor.get_avg_k()*floor.get_total_area()

        if self.get_total_area() == 0:
            self._avg_k = 0
            return self._avg_k

        self._avg_k = round(self._avg_k/self.get_total_area(), 4)
        return self._avg_k

    def __str__(self):
        return f"Building name:{self._name}, floor num:{self._floor_num}, house num:{self._house_num}, floors:{self._floors}"

    def __repr__(self):
        return self.__str__()

    def get_floor_tensor(self, floor_id):

        for floor in self._floors:
            if floor.get_flood_id() == floor_id:
                return floor.to_tensor()
        return None, None

    def get_floor(self, floor_id):
        for floor in self._floors:
            if floor.get_flood_id() == floor_id:
                return floor

        return None

    def get_cost_view(self):  # 获取成本视图

        # pdb.set_trace()

        self._cost_view = {}
        for floor in self._floors:
            for key in floor.get_cost_view():

                assert key is not None and key != '', f"key is None or ''"

                # print("floor key ", key)
                if key is None or key == '':
                    continue

                if key in self._cost_view:
                    self._cost_view[key]['area'] += round(
                        float(floor.get_cost_view()[key]['area']), 4)
                    self._cost_view[key]['cost'] += round(
                        float(floor.get_cost_view()[key]['cost']), 2)
                else:
                    # self._cost_view.append(key)
                    self._cost_view[key] = {
                        'area': round(floor.get_cost_view()[key]['area'], 4),
                        'price': round(floor.get_cost_view()[key]['price'], 2),
                        'cost': round(floor.get_cost_view()[key]['cost'], 4)
                    }

        return self._cost_view

    view_url = "http://apoco:apoco2023@couchdb.apoco.com.cn/design_mate/_design/materials_view/_view/materials_view?"

    def get_cost_view_from_couchdb(self, view_url=None):  # 获取成本视图,从couchdb中获取

        if view_url is not None:
            self.view_url = view_url

        self._cost_view = {}

        total_cost = 0

        #url = f"{self.view_url}?id=\"{self.couchdb_doc_id}\""
        url = f"{self.view_url}startkey=%5B%22{self.couchdb_doc_id}%22%5D&endkey=%5B%22{self.couchdb_doc_id}%22%2C%7B%7D%5D"

        print("url ", url)

        response = requests.get(url)
        data = response.json()
        i = 0
        for row in data.get("rows", []):

            key = row['key'][1]
            value = row['value']

            if key in self._cost_view:
                area = round(float(self._cost_view[key]['area'] + value['area']), 4)
                cost = round(float(self._cost_view[key]['cost'] + value['cost']), 2)
                self._cost_view[key]['area'] = area
                self._cost_view[key]['cost'] = cost

            else:
                self._cost_view[key] = {
                    'area': round(float(value['area']), 4),
                    'price': round(float(value['price']), 2),
                    'cost': round(float(value['cost']), 2)
                }
            total_cost += round(float(value['cost']), 2)

        self._cost_view['total'] = {
            'area': None,
            'price': None,
            'cost': round(total_cost,2)
        }

        return self._cost_view

    def to_json(self, s_id):
        return {
            "_id": s_id,  # 用于 couchdb中的 doc_id,也存入 sqlite db中，通过building_id查询
            "created_time": self.create_time,  # 创建时间
            "building_name": self._name,
            "building_id": self._id,
            "floor_num": self._floor_num,
            "house_num": self.get_house_num(),
            "form_factor": self.get_form_factor(),  # 建筑形态系数
            "building_area": self.get_total_area(),  # 总面积
            "building_out_wall_area": self.get_out_wall_area(),  # 外墙面积
            "building_total_window_area": self.get_total_window_area(),      # 总窗面积
            # "building_total_glass_area": self.get_total_glass_area(),  # 总玻璃面积
            # 总造价
            "building_total_cost": round(float(self.get_total_cost()), 4),
            # 平均窗墙比
            "building_avg_ww_ratio": round(float(self.get_avg_ww_ratio()), 4),
            "building_avg_k": round(float(self.get_avg_k()), 4),  # 平均K值
            "floors": [floor.to_json() for floor in self._floors]  # 楼层列表及其信息

        }

    def to_json_cn(self, s_id):
        return {
            "_id": s_id,  # 用于 couchdb中的 doc_id,也存入 sqlite db中，通过building_id查询
            "created_time(创建时间)": self.create_time,  # 创建时间
            "building_name(建筑名称)": self._name,
            "building_id(建筑id)": self._id,
            "floor_num(楼层数)": self._floor_num,
            "house_num(户数)": self.get_house_num(),
            "form_factor(体型系数)": self.get_form_factor(),  # 建筑形态系数
            "building_area(建筑面积)": self.get_total_area(),
            "building_out_wall_area(外墙面积)": self.get_out_wall_area(),
            "building_total_window_area(总窗面积)": self.get_total_window_area(),
            # "building_total_glass_area(总玻璃面积)": self.get_total_glass_area(),
            "building_total_cost(总造价)": round(float(self.get_total_cost()), 4),
            "building_avg_ww_ratio(平均窗墙比)": round(float(self.get_avg_ww_ratio()), 4),
            "building_avg_k(平均K值)": round(float(self.get_avg_k()), 4),
            "floors(楼层)": [floor.to_json_cn() for floor in self._floors]
        }

    def json_to_building(self, json_data):

        self._name = json_data['building_name']
        self._id = json_data["building_id"]
        self.couchdb_doc_id = json_data["_id"]
        self.create_time = json_data["created_time"]
        self._floor_num = json_data["floor_num"]
        self._house_num = json_data["house_num"]
        self._form_factor = json_data["form_factor"]
        self._total_area = json_data["building_area"]
        self._out_wall_area = json_data["building_out_wall_area"]
        self._total_window_area = json_data["building_total_window_area"]

        self._floors = [Floor().json_to_floor(floor)
                        for floor in json_data["floors"]]
        

        return self


    def calaculate_all(self):
        
        self._area = 0
        self._out_wall_area = 0
        self._total_window_area = 0
        self._total_glass_area = 0
        self._total_cost = 0
        self._avg_ww_ratio = 0
        self._avg_k = 0

        t_k = 0

        for floor in self._floors:
            floor.calculate_all()

            self._area += floor.get_area()
            self._out_wall_area += floor.get_out_wall_area()
            self._total_window_area += floor.get_total_window_area()
            #self._total_glass_area += floor.get_total_glass_area()
            self._total_cost += floor.get_total_cost()
            t_k += floor.get_avg_k()*floor.get_area()
            
        self._avg_ww_ratio = self._total_window_area / self._out_wall_area
        self._avg_k = round(t_k / self._area,4)
        self._form_factor = self.get_form_factor()


    def tensor_to_building(self, floors_tensors):

        for floor_index, floor in enumerate(self.get_floors()):

            floor = floor.tensor_to_floor(floors_tensors)
            self.update_floor(copy.deepcopy(floor), floor_index)

        self.calaculate_all()

        return self

class BuildingCreator:

    houses = []
    staircases = []
    corridors = []

    def __init__(self, floor_num, house_num, staircase_num=1, corridor_num=0):
        self.floor_num = floor_num
        self.house_num = house_num
        self.staircase_num = staircase_num
        self.corridor_num = corridor_num

    def create_building(self, buildingName=None, houses=[], staircases=[], corridors=[]):

        #

        building = Building(id=0, name=buildingName, area=0, out_wall_area=0,
                            floor_num=self.floor_num, house_num=self.house_num)
        #

        for index in range(self.floor_num):

            temp_houses = copy.deepcopy(houses)
            temp_staircases = copy.deepcopy(staircases)
            temp_corridors = copy.deepcopy(corridors)

            # 修改houses 中的house name，添加floor name
            for h_index in range(len(houses)):
                temp_houses[h_index].set_name(
                    f'floor_{index}/{temp_houses[h_index].get_name()}')

            floor = Floor(name=f'floor_{index}',
                          floor_id=index, houses=temp_houses,
                          staircases=temp_staircases, corridors=temp_corridors)

            building.add_floor(copy.deepcopy(floor))

        return building


def save_to_couchdb(building):
    try:
        doc_id = generate_random_string()
        doc = building.to_json(doc_id)
        doc = json.loads(json.dumps(doc, ensure_ascii=False,
                         indent=4, cls=Floor.CustomJSONEncoder))
        couchdb_pool.insert_doc(doc)
    except Exception as e:
        print(e)
        return None
    return doc_id


def generate_random_string():
    timestamp = int(time.time() * 1000)  # 获取当前时间戳并转换为毫秒
    sequence = random.randint(0, 9999)  # 生成一个随机的序列号，可以根据需要调整范围
    unique_string = f"{timestamp:013d}{sequence:04d}"
    return unique_string

# 随机生成每层楼允许的 house_num, staircase_num, corridor_num


def generate_space_numbers():

    HOUSE_NUMBER = 12
    STAIRCASE_NUMBER = 4

    # 生成随机的 house_num
    house_num = random.randint(6, HOUSE_NUMBER)

    # 生成随机的 staircase_num，满足要求
    max_staircase = min(STAIRCASE_NUMBER, house_num // 4)
    staircase_num = random.randint(1, max_staircase)

    # 生成随机的 corridor_num，根据 house_num 决定概率
    if house_num > 8:
        corridor_num = random.choice([0, 2, 4])
    else:
        corridor_num = random.choice([0, 2])

    return house_num, staircase_num, corridor_num

def extract_building_features(building_features):

    #pdb.set_trace()

    # 判断是否为numpy数组，如果不是则转换为numpy数组
    # if not isinstance(building_features[0], np.ndarray):
    building_features = np.array(building_features[0])

    position_0 = 0
    position_1 = House.HOUSE_FEATURES * Floor.HOUSE_NUMBER
    position_2 = position_1 + PublicSpace.ROOM_FEATURES_LEN * Floor.STAIRCASE_NUMBER
    position_3 = position_2 + PublicSpace.ROOM_FEATURES_LEN * Floor.CORRIDORS_NUMBER

    # copy house_features from building_features
    house_features = building_features[position_0:position_1]
    # copy stair_features from building_features
    stair_features = building_features[position_1:position_2]
    # copy corridor_features from building_features
    corridor_features = building_features[position_2:position_3]

    return house_features, stair_features, corridor_features


def main(args):

    #pdb.set_trace()

    start_time = time.time()
    
    floor_num = 2
    # floor_house_num = 12

    house_num, staircase_num, corridor_num = generate_space_numbers()

    hc = HousesCreator(house_num)  # 生成户型
    houses = hc.make_houses()

    sc = publicSpaceCreator(staircase_num, space_code=12)  # 生成楼梯
    staircases = sc.make_public_spaces()

    cc = publicSpaceCreator(corridor_num, space_code=13)  # 生成走廊
    corridors = cc.make_public_spaces()

    bc = BuildingCreator(floor_num, house_num)
    building = bc.create_building(buildingName="building", houses=houses,
                                  staircases=staircases, corridors=corridors)

    doc_id = save_to_couchdb(building)
    building.set_doc_id(doc_id)

    if doc_id:
        print(f"save to couchdb success {doc_id}")
    else:
        print("save to couchdb failed")

    # print(building.get_floor_tensor(2))

    #print(building.get_cost_view_from_couchdb())
    print(json.dumps(building.get_cost_view_from_couchdb(), indent=2, ensure_ascii=False))

    data_set_x,data_set_y = building.get_floor_tensor(1)

    append_to_dataset(f'{args.datasets_dir}/data_set_x.pkl', data_set_x)
    append_to_dataset(f'{args.datasets_dir}/data_set_y.pkl', data_set_y)

    end_time = time.time()

    print(f"cost time {end_time - start_time} s")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Dataset Generation and Storage")
    parser.add_argument("--datasets_dir", type=str, required=True,
                        default="./datasets", help="dataset directory")
    args = parser.parse_args()
    main(args)