import pdb
import tensorflow as tf
import json
import random
import numpy as np
import copy
import os
import time
from apocolib.GPUConfigurator import GPUConfigurator
from ml_server_v2.WindowOptimization import WindowOptimization
from ml_server_v2.SpaceStandard import SpaceStandard
from apocolib.MlLogger import mlLogger as ml_logger
from ml_server_v2.MaterialWarehouse import GlassMaterialWarehouse, WallInsulationMaterialWarehouse, WindowFrameMaterialWarehouse
import sys
sys.path.append("..")


current_path = os.getcwd()
# print('current_path', current_path)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# tf.config.set_visible_devices([], 'GPU')
# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将 ml_server 目录路径添加到模块搜索路径中
sys.path.append(current_dir)

# configurator = GPUConfigurator(use_gpu=False, gpu_memory_limit=None)
configurator = GPUConfigurator(use_gpu=True, gpu_memory_limit=1024)
configurator.configure_gpu()
tf.device(configurator.select_device())
#tf.config.run_functions_eagerly(True)

gmWarehouse = GlassMaterialWarehouse()
wimWarehouse = WallInsulationMaterialWarehouse()
wfmWarehouse = WindowFrameMaterialWarehouse()
_ss = SpaceStandard()  # 创建空间标准对象


# 定义材料基础类
class Material:
    def __init__(self, area=0.0, material_type=None, warehouse=None, price=0.0, K=0.0):
        self._area = round(area, 4)
        self._material_type = material_type
        self.warehouse = warehouse
        self.price = round(price, 2)
        self.K = round(K, 4)

    def get_area(self):
        return self._area

    def set_area(self, area):
        self._area = round(area, 4)

    def get_material_type(self):
        return self._material_type

    def set_material_type(self, material_type):
        self._material_type = material_type

    def set_warehouse(self, warehouse):
        self.warehouse = warehouse

    def get_warehouse(self):
        return self.warehouse

    def get_cost(self):
        # print('material_type:',self._material_type,'price:',self.price,'area:',self._area)
        return round(self._area * self.price, 4)

    def get_K(self):
        return self.K

    def get_price(self):
        return self.price

    def set_price(self, price):
        self.price = round(price, 2)

    # 生成独热编码
    def get_one_hot_code(self):
        encoding = [0] * self.warehouse.get_size()
        # print('encoding:',encoding)
        # print('material_type:',self._material_type)
        # print('key = ',self.warehouse.find_key(self._material_type))

        if self._material_type:
            key = self.warehouse.find_key(self._material_type)
            if key is not None:
                encoding[int(key)] = 1
        # print('encoding:',encoding)
        # to tensor
        encoding = tf.convert_to_tensor(encoding, dtype=tf.float64)

        return encoding

    # 根据独热编码生成材料类型
    def get_material_type_by_one_hot_code(self, encoding):
        key = np.argmax(encoding)
        # print("encoding ", encoding, " key ", key)
        # return key
        return self.warehouse.get_material(key)

    def get_cost_view(self):

        self._cost_view = {}
        self._cost_view[self.get_material_type] = {
            "area": self._area,
            "price": self.price,
            "cost": self.get_cost()
        }
        return self._cost_view


class Glass(Material):
    def __init__(self, area=0.0, width=0.0, height=0.0, material_type=None, warehouse=None, price=0.0, K=0.0):

        if material_type is not None:
            price = material_type['price']
            K = material_type['K']

        Material.__init__(self, area, material_type, warehouse, price, K)
        self._width = round(width, 4)
        self._height = round(height, 4)

    def get_width(self):
        return self._width

    def get_height(self):
        return self._height

    def get_cost(self):
        return super().get_cost()


class WindowFrame(Material):
    def __init__(self, area=0.0, length=0.0, material_type=None, warehouse=None, price=0.0, K=0.0):

        if material_type is not None:
            price = material_type['price']
            K = material_type['K']

        super().__init__(area, material_type, warehouse, price, K)
        self._length = length

    def get_length(self):
        return self._length

    def set_length(self, length):
        self._length = round(length, 4)

    def get_cost(self):
        return super().get_cost()


class WallInsulation(Material):
    def __init__(self, area=0.0, thickness=0.0, material_type=None, warehouse=None, price=0.0, K=0.0):

        if material_type is not None:
            price = material_type['price']
            K = material_type['K']

        super().__init__(area, material_type, warehouse, price, K)
        self._thickness = round(thickness, 2)

    def get_thickness(self):
        return self._thickness

    def set_thickness(self, thickness):
        self._thickness = round(thickness, 2)

    def get_cost(self):
        return super().get_cost()


class Orientation:

    EAST = 0
    SOUTH = 1
    WEST = 2
    NORTH = 3

    def __init__(self):
        # 朝向
        self.orientation_mapping = {
            self.EAST: '东',
            self.SOUTH: '南',
            self.WEST: '西',
            self.NORTH: '北'
        }

    def get_orientation_OHE(self, orientation) -> np.ndarray:
        one_hot_encoding = np.zeros(len(self.orientation_mapping))
        one_hot_encoding[orientation] = 1
        return one_hot_encoding

    def get_orientation_OHL(self) -> np.ndarray:
        one_hot_list = []
        for i in range(len(self.orientation_mapping)):
            one_hot_list.append(self.get_orientation_OHE(i))
        return one_hot_list

    # print(get_orientation_OHL())
    # 查询所有朝向的key
    # @staticmethod
    def get_orientation_keys(self):
        return list(self.orientation_mapping.keys())

    # 查询朝向数量
    # @staticmethod
    def get_o_num(self) -> int:
        return len(self.orientation_mapping)

    # 查询朝向
    # @staticmethod
    def get_orientation(self, orientation):
        for key, value in self.orientation_mapping.items():
            if value == orientation:
                return key
        return None

    def get_orientation(self, orientation):
        return self.orientation_mapping[orientation]

    def get_orientation_mapping(self):
        return self.orientation_mapping


or_instance = Orientation()


class Window:

    WINDOW_FEATURES_LEN = 2 + gmWarehouse.get_size() + wfmWarehouse.get_size()

    def __init__(self, area=0.0, width=0.0, height=0.0, glass=None, window_frame=None, orientation=None):
        self._area = round(area, 4)
        self._width = round(width, 4)
        self._height = round(height, 4)
        self._glass = glass
        self._window_frame = window_frame
        self._orientation = orientation

        # self.WINDOW_FEATURES_LEN = self.get_window_features_len()

    def get_area(self):
        self._area = round(self._width * self._height, 4)
        return self._area

    def get_width(self):
        return self._width

    def set_width(self, width):
        self._width = round(width, 4)

    def get_height(self):
        return self._height

    def set_height(self, height):
        self._height = round(height, 4)

    def get_glass(self):
        return self._glass

    def set_glass(self, glass):
        self._glass = glass

    def get_window_frame(self):
        return self._window_frame

    def set_window_frame(self, window_frame):
        self._window_frame = window_frame

    def get_orientation(self):
        return self._orientation

    def set_orientation(self, orientation):
        self._orientation = orientation

    def get_cost(self):
        return round(self._glass.get_cost() + self._window_frame.get_cost(), 4)

    def get_cost_view(self):

        self._cost_view = {}
        self._cost_view.append(self._glass.get_cost_view())
        self._cost_view.append(self._window_frame.get_cost_view())

        return self._cost_view

    # 计算窗户的平均K值,单位为W/(m2*K),即窗户的U值,计算公式为:U=(K1*S1+K2*S2)/S1+S2
    def get_avg_k(self):

        if self._glass.get_area() == 0 or self._window_frame.get_area() == 0:
            return 0

        return round((self._glass.get_K() * self._glass.get_area() + self._window_frame.get_K() * self._window_frame.get_area()) / (self._glass.get_area() + self._window_frame.get_area()), 4)

    # 计算窗框/窗总面积比值,计算公式为:R=S1/S2+S1
    def get_fw_ratio(self):

        if self._glass.get_area() == 0 or self._window_frame.get_area() == 0:
            return 0

        return round(self._window_frame.get_area() / (self._glass.get_area() + self._window_frame.get_area()), 4)

    # to_tensor
    def to_tensor(self):  # shape (24,)

        area_tensor = tf.cast(tf.constant(
            [self._area, self._orientation]), dtype=tf.float64)
        glass_tensor = tf.cast(tf.constant(
            self.get_glass().get_one_hot_code()), dtype=tf.float64)
        window_frame_tensor = tf.cast(tf.constant(
            self.get_window_frame().get_one_hot_code()), dtype=tf.float64)
        tensor = tf.concat(
            [area_tensor, glass_tensor, window_frame_tensor], axis=0)
        # 数据类型转换为float64
        tensor = tf.cast(tensor, tf.float64)

        return tensor

    # tensor to window, shape (24,)
    def tensor_to_window(self, tensor,wall_width=0.0,wall_height=0.0):

        def split_window_tensor(tensor):

            if len(tensor) != Window.WINDOW_FEATURES_LEN:

                ml_logger.error("The length of tensor is not equal to the length of window features.")
                return None
            
            area = tensor[0]

            if area == 0.0:
                ml_logger.warning("The area of window is 0.0.")
                return 0,0,None,None

            orientation = int(tensor[1])
            glass_tensor = tensor[2:2+gmWarehouse.get_size()]
            window_frame_tensor = tensor[2+gmWarehouse.get_size():2+gmWarehouse.get_size()+wfmWarehouse.get_size()]

            #ml_logger.info("The area of window is %f, the orientation of window is %s, the glass tensor is %s, the window frame tensor is %s." % (
            #    area, orientation, glass_tensor, window_frame_tensor))

            return area, orientation,glass_tensor, window_frame_tensor

        self._area, self._orientation, glass_tensor, window_frame_tensor = split_window_tensor(tensor)

        self._width, self._height = calculate_window_dimensions(self._area,wall_width,wall_height)

        if self._area == 0.0:
            ml_logger.warning("The area of window is 0.0.")
            return self

        wfa_ratio, _ = wfmWarehouse.get_best_wf(self._width,self._height)


        self._glass = Glass(
                            area=self._area*(1-wfa_ratio),
                            warehouse=gmWarehouse,
                            material_type=Glass(warehouse=gmWarehouse).get_material_type_by_one_hot_code(glass_tensor))

        self._window_frame = WindowFrame(
                            area=self._area*wfa_ratio,
                            warehouse=wfmWarehouse,
                            material_type=WindowFrame(warehouse=wfmWarehouse).get_material_type_by_one_hot_code(window_frame_tensor))
        #self._area = tensor[0]
        #self._orientation = tensor[1]
        # glass = Glass(warehouse=gmWarehouse,
        #               material_type=Glass(warehouse=gmWarehouse).get_material_type_by_one_hot_code(tensor[2:2+gmWarehouse.get_size()]))
        # self._glass = glass

        # window_frame = WindowFrame(warehouse=wfmWarehouse,
        #                            material_type=WindowFrame(warehouse=wfmWarehouse).get_material_type_by_one_hot_code(tensor[2+gmWarehouse.get_size():2+gmWarehouse.get_size()+wfmWarehouse.get_size()]))
        # self._window_frame = window_frame
        # window = Window(area=area, glass=glass, window_frame=window_frame, orientation=orientation)

        return self

    def get_empty_tensor():
        tensor = tf.constant([0.0, 0] +
                             [0] * gmWarehouse.get_size() +
                             [0] * wfmWarehouse.get_size())
        # 数据类型转换为float64
        tensor = tf.cast(tensor, tf.float64)

        return tensor

    def get_window_features_len(self):
        return 2 + gmWarehouse.get_size() + wfmWarehouse.get_size()


class Wall:

    WALL_FEATURES_LEN = 2 + wimWarehouse.get_size() + 2 + gmWarehouse.get_size() + \
        wfmWarehouse.get_size()

    def __init__(self, area=0.0, width=0.0, height=2.95, thickness=0.0, wi_material=None, orientation=0, window=None):
        # thickness为墙体厚度,material为墙体材料,orientation为墙体朝向,如北墙,南墙等,
        self._area = round(area, 4)
        self._width = round(width, 4)
        self._height = round(height, 4)
        self._thickness = round(thickness, 4)
        self._material = wi_material
        self._orientation = orientation
        self._window = window
        if self._window is not None:
            self.add_window(window)

        # 墙体特征长度，
        # self.wall_features_len = 2 + wimWarehouse.get_size() + 2 + gmWarehouse.get_size() + \
        #     wfmWarehouse.get_size()

        # self.WALL_FEATURES_LEN = self.get_wall_features_len()

        self.calculated = False

    def add_window(self, window):
        if self._window is None:
            if self.is_window_valid(window):
                self._window = window

                # 修正外墙材料的面积
                self._material.set_area(self.get_area() - window.get_area())

        else:
            print("This wall already has a window.")

    def reset_window(self, window):
        if self._window is not None:
            if self.is_window_valid(window):
                self._window = window

                # 修正外墙材料的面积
                self._material.set_area(self.get_area() - window.get_area())
        else:
            print("This wall has no window.")

    def reset_im_material_area(self):
        self._material.set_area(self.get_area() - self._window.get_area())

    # 判断窗体的数据是否合法
    def is_window_valid(self, window):
        if window.get_width() > self._width:
            raise self.WallDataError(
                "The width of the window is greater than the length of the wall.")

        if window.get_height() > self._height:
            raise self.WallDataError(
                "The height of the window is greater than the height of the wall.")

        if window.get_orientation() != self._orientation:
            raise self.WallDataError(
                "The orientation of the window is not the same as the orientation of the wall.")
        return True

    def remove_window(self):
        if self._window is not None:
            self._window = None
        else:
            print("There is no window on this wall.")

    def calculate_area(self):
        self._area = round(self._width * self._height, 4)
        return self._area

    def get_area(self):
        return self.calculate_area()

    def get_width(self):
        return self._width

    def set_width(self, width):
        self._width = round(width, 4)

    def get_height(self):
        return self._height

    def set_height(self, height):
        self._height = round(height, 4)

    def get_material(self):
        return self._material

    def set_material(self, material):
        self._material = material

    def get_material_area(self):
        if self._material is not None:
            return self._material.get_area()
        else:
            return 0.0

    def get_material_cost(self):
        if self._material is not None:
            return self._material.get_cost()
        else:
            return 0.0

    def get_glass_cost(self):
        if self._window is None:
            return 0
        return self._window.get_glass().get_cost()

    def get_frame_cost(self):
        if self._window is None:
            return 0
        return self._window.get_window_frame().get_cost()

    def get_orientation(self):
        return self._orientation

    def set_orientation(self, orientation):
        self._orientation = orientation

    def get_window(self):
        return self._window

    def get_cost(self):  # 计算墙体成本,单位为元,计算公式为:墙体材料成本 + 窗户成本（如果有窗户，含玻璃和窗框
        if self._window is None:
            if self._material is not None:
                return self._material.get_cost()
            else:
                return 0.0

        return round(self._material.get_cost() + self._window.get_cost(), 4)

    def get_cost_view(self):
        self._cost_view = {}
        if self._window is not None:
            self._cost_view.append(self._window.get_cost_view())
        if self._material is not None:
            self._cost_view.append(self._material.get_cost_view())

        return self._cost_view

    def get_window_area(self):
        if self._window is None:
            return 0.0
        return self._window.get_area()

    def get_avg_k(self):  # 计算墙体的平均K值,单位为W/(m2*K),计算公式为:U=(K1*S1+K2*S2)/S1+S2；其中K1为墙体材料的K值，S1为墙体面积，K2为窗户的K值，S2为窗户面积
        if self._window is None:
            if self._material is not None:
                return self._material.get_K()
            else:
                return 0.0

        if self.calculate_area() == 0:
            return 0.0

        return round((self._material.get_K() * (self.calculate_area()-self._window.get_area()) + self._window.get_avg_k() * self._window.get_area()) / self.calculate_area(), 4)

    def get_ww_ratio(self):  # 计算墙/窗比值,计算公式为:R=S2/S1；其中S1为墙体面积，S2为窗户面积

        if self._window is None:
            return 0.0

        if self.calculate_area() == 0:
            return 0.0

        return round(self._window.get_area() / self.calculate_area(), 4)

    or_instance = Orientation()


    def calulate_all(self):

        self._total_cost = 0.0
        self._material_area = 0.0
        self._window_area = 0.0
        self._avg_k = 0.0

        if self._material is not None and isinstance(self._material, Material):
            self._material_area = self._material.get_area()
            self._total_cost += self._material.get_cost()

        if self._window is not None and isinstance(self._window, Window):
            self._window_area = self._window.get_area()
            self._total_cost += self._window.get_cost()

            self._avg_k = round((self._material.get_K() * (self.calculate_area()-self._window.get_area()) + self._window.get_avg_k() * self._window.get_area()) / self.calculate_area(), 4)

        else:
            self._avg_k = self._material.get_K()

        self.calculated = True

 
    def to_json(self):

        # pdb.set_trace()

        if self._window is None:
            return {
                # 墙体朝向
                # or_instance.get_orientation(self._orientation),
                "orientation": self._orientation,
                "area": self._area,  # 墙体面积
                "wall_width": self._width,  # 墙体宽度
                "wall_height": self._height,  # 墙体高度
                "insulation_material": self._material.get_material_type(),  # 墙体材料
                "material_area": self.get_material_area(),
                "wall_avg_k": self.get_avg_k(),
                "window": None
            }
        else:
            wf_material = self._window.get_window_frame().get_material_type()
            wf_m_value = {
                'type': wf_material['type'],
                'material': wf_material['material'],
                'price': wf_material['price'],
                'K': wf_material['K']
            }

            # wf_m_value = {'key': wf_material['key'], 'type': wf_material['type'],
            #               'material': wf_material['material'], 'price': wf_material['price'], 'K': wf_material['K']}
            return {
                # 墙体朝向
                # or_instance.get_orientation(self._orientation),
                "orientation": self._orientation,
                "area": self._area,  # 墙体面积
                "wall_width": self._width,  # 墙体宽度
                "wall_height": self._height,  # 墙体高度
                "insulation_material": self._material.get_material_type(),  # 墙体材料
                "material_area": self.get_material_area(),
                "wall_avg_k": self.get_avg_k(),
                "window": {
                    "area": self._window.get_area(),  # 窗户面积
                    "window_wall_ratio": self.get_ww_ratio(),  # 窗/墙比值
                    "window_width": self._window.get_width(),  # 窗户宽度
                    "window_height": self._window.get_height(),  # 窗户高度
                    # 窗户朝向
                    # or_instance.get_orientation(self._window.get_orientation()),
                    "orientation": self._window.get_orientation(),
                    "glass_material": self._window.get_glass().get_material_type(),  # 窗户玻璃材料
                    "wfa_ratio": self._window.get_fw_ratio(),  # 窗框/窗比值
                    "glass_area": self._window.get_glass().get_area(),  # 玻璃面积
                    # "wf_material": self._window.get_window_frame().get_material_type(),#窗框材料
                    "wf_material": wf_m_value,  # 窗框材料
                    "wf_area": self._window.get_window_frame().get_area()  # 窗框面积
                }
            }

    # json to wall, 从json数据中读取数据，创建墙体对象

    def json_to_wall(self, json_data):
        self._orientation = json_data["orientation"]
        self._area = json_data["area"]
        self._width = json_data["wall_width"]
        self._height = json_data["wall_height"]
        self._material = WallInsulation(area=round(
            json_data["area"], 4), warehouse=wimWarehouse, material_type=json_data["insulation_material"])
        # self._material.set_material_type(json_data["insulation_material"])
        if json_data["window"] is not None:
            self._material.set_area(
                self._material.get_area()-json_data["window"]["area"])
            self._window = Window(area=json_data["window"]["area"])
            # self.reset_im_material_area()
            # self.add_window(Window(area=json_data["window"]["area"]))
            # self._window.set_area(json_data["window"]["area"])
            self._window.set_orientation(json_data["window"]["orientation"])
            self._window.set_width(json_data["window"]["window_width"])
            self._window.set_height(json_data["window"]["window_height"])
            self._window.set_glass(Glass(area=json_data["window"]["glass_area"], warehouse=gmWarehouse,
                                         material_type=json_data["window"]["glass_material"]))
            # self._window.get_glass().set_material_type(json_data["window"]["glass_material"])
            self._window.set_window_frame(WindowFrame(area=json_data["window"]["wf_area"], warehouse=wfmWarehouse,
                                                      material_type=json_data["window"]["wf_material"]))
            # self._window.get_window_frame().set_material_type(json_data["window"]["wf_material"])
        else:
            self._window = None

        return self

    def to_json_cn(self):
        if self._window is None:
            return {
                # 墙体朝向
                # or_instance.get_orientation(self._orientation),
                # self._orientation,
                "orientation(墙体朝向)": or_instance.get_orientation(self._orientation),
                "area(墙体面积)": self._area,  # 墙体面积
                "wall_width(墙体宽度)": self._width,  # 墙体宽度
                "wall_height(墙体高度)": self._height,  # 墙体高度
                # 墙体材料
                "insulation_material(墙体材料)": self._material.get_material_type() if self._material is not None else {},
                "material_area(材料面积)": self.get_material_area(),  # 材料面积
                "wall_avg_k(墙体平均导热系数)": self.get_avg_k(),  # 墙体平均导热系数
                "window": None
            }
        else:
            wf_material = self._window.get_window_frame().get_material_type()

            wf_m_value = {'type': wf_material['type'],
                          'material': wf_material['material'], 'price': wf_material['price'], 'K': wf_material['K']}
            # wf_m_value = {'key': wf_material['key'], 'type': wf_material['type'],
            #               'material': wf_material['material'], 'price': wf_material['price'], 'K': wf_material['K']}
            return {
                # 墙体朝向
                # or_instance.get_orientation(self._orientation),
                # self._orientation,
                "orientation(墙体朝向)": or_instance.get_orientation(self._orientation),
                "area(墙体面积)": self._area,  # 墙体面积
                "wall_width(墙体宽度)": self._width,  # 墙体宽度
                "wall_height(墙体高度)": self._height,  # 墙体高度
                # 墙体材料
                "insulation_material(墙体材料)": self._material.get_material_type(),
                "material_area(材料面积)": self.get_material_area(),  # 材料面积
                "wall_avg_k(墙体平均导热系数)": self.get_avg_k(),  # 墙体平均导热系数
                "window": {
                    "area(窗户面积)": self._window.get_area(),  # 窗户面积
                    "window_wall_ratio(窗/墙比)": self.get_ww_ratio(),  # 窗/墙比值
                    "window_width(窗户宽度)": self._window.get_width(),  # 窗户宽度
                    "window_height(窗户高度)": self._window.get_height(),  # 窗户高度
                    # 窗户朝向
                    # or_instance.get_orientation(self._window.get_orientation()),
                    # self._window.get_orientation(),
                    "orientation(窗户朝向)": or_instance.get_orientation(self._window.get_orientation()),
                    # 窗户玻璃材料
                    "glass_material(窗户玻璃材料)": self._window.get_glass().get_material_type(),
                    "wfa_ratio(窗框/窗比)": self._window.get_fw_ratio(),  # 窗框/窗比值
                    # 玻璃面积
                    "glass_area(玻璃面积)": self._window.get_glass().get_area(),
                    # "wf_material(窗框材料)": self._window.get_window_frame().get_material_type(),#窗框材料
                    "wf_material(窗框材料)": wf_m_value,
                    # 窗框面积
                    "wf_area(窗框面积)": self._window.get_window_frame().get_area()
                }
            }

    # to tensor

    def to_tensor(self):  # shape=(34,)

        if self._window is None:  # 是外墙，但是无窗户，只有墙体材料，需要补充一个空的窗户
            area_tensor = tf.cast(tf.constant(
                [self._area, self._orientation]), dtype=tf.float64)
            material_tensor = tf.cast(tf.constant(
                self._material.get_one_hot_code()), dtype=tf.float64)
            window_tensor = tf.cast(
                Window.get_empty_tensor(), dtype=tf.float64)

            # print("Wall area tensor: ", area_tensor,"shape: ", area_tensor.shape)
            # print("Wall material tensor: ", material_tensor,"shape: ", material_tensor.shape)
            # print("Wall empty window tensor: ", window_tensor, "shape: ", window_tensor.shape)

            tensor = tf.concat(
                [area_tensor, material_tensor, window_tensor], axis=0)

        else:  # 是外墙，有窗户
            area_tensor = tf.cast(tf.constant(
                [self._area, self._orientation]), dtype=tf.float64)
            material_tensor = tf.cast(tf.constant(
                self._material.get_one_hot_code()), dtype=tf.float64)
            window_tensor = self.get_window().to_tensor()
            window_tensor = tf.cast(window_tensor, dtype=tf.float64)

            # print("Wall area tensor: ", area_tensor,"shape: ", area_tensor.shape)

            # print("Wall material tensor: ", material_tensor,"shape: ", material_tensor.shape)
            # print("Wall window tensor: ", window_tensor, "shape: ", window_tensor.shape)

            tensor = tf.concat(
                [area_tensor, material_tensor, window_tensor], axis=0)

        #ml_logger.info(
        #    f"Convert Wall {self._orientation} to tensor,shape = {tensor.shape}")

        _,_,_,_ = self.split_wall_tensor(tensor)
        
        # print("wall tensor: ", tensor)
        return tensor

    def split_wall_tensor(self,tensor):

        if len(tensor) != Wall.WALL_FEATURES_LEN :
            ml_logger.error(
                f"Wall tensor length {len(tensor)} is not equal to {Wall.WALL_FEATURES_LEN}")
            return None
        
        #print("Wall tensor: ", tensor)

        area = tensor[0]

        if area == 0:
            #ml_logger.warning("Wall area is 0")
            return 0, 0, None, None

        orientation = int(tensor[1])  # 强制换行为int
        material_tensor = tensor[2:2+wimWarehouse.get_size()]
        window_tensor = tensor[2+wimWarehouse.get_size():]

        #ml_logger.info(
        #    f"Split wall tensor, area {area}, orientation {orientation}, material_tensor {material_tensor}, window_tensor {window_tensor}")

        return area, orientation, material_tensor, window_tensor

    # tensor to wall, 从tensor中恢复墙体信息
    # , wimWarehouse, gmWarehouse, wfmWarehouse):
    def tensor_to_wall(self, tensor):
        
        #pdb.set_trace()

        # def split_wall_tensor(tensor):

        #     if len(tensor) != Wall.WALL_FEATURES_LEN :
        #         ml_logger.error(
        #             f"Wall tensor length {len(tensor)} is not equal to {Wall.WALL_FEATURES_LEN}")
        #         return None
            
        #     print("Wall tensor: ", tensor)

        #     area = tensor[0]

        #     if area == 0:
        #         ml_logger.warning("Wall area is 0")
        #         return 0, 0, None, None

        #     orientation = int(tensor[1])  # 强制换行为int
        #     material_tensor = tensor[2:2+wimWarehouse.get_size()]
        #     window_tensor = tensor[2+wimWarehouse.get_size():]

        #     ml_logger.info(
        #         f"Split wall tensor, area {area}, orientation {orientation}, material_tensor {material_tensor}, window_tensor {window_tensor}")

        #     return area, orientation, material_tensor, window_tensor

        #area, self._orientation, material_tensor, window_tensor = split_wall_tensor(tensor)

        area, _, material_tensor, window_tensor = self.split_wall_tensor(tensor)

        #ml_logger.info(f"-------  self._area {self._area}, area in wall_tenosor {area} ")

        # if area != self._area:
        #     ml_logger.warning(
        #         f"------ Wall area {area} is not equal to {self._area}")
            #return None

        #self._area = area
        #self._area = tensor[0]
        if self._area == 0:
            ml_logger.warning("Wall area is 0, return self")
            return self


        self._window.tensor_to_window(window_tensor,wall_width=self._width,wall_height=self._height)

        self._material = WallInsulation(
                material_type=WallInsulation(warehouse=wimWarehouse).get_material_type_by_one_hot_code(material_tensor))

        self._material.set_area(self._area - self._window.get_area())

        #self._orientation = int(tensor[1])  # 强制换行为int
        # self._material = WallInsulation(material_type=WallInsulation(
        #     warehouse=wimWarehouse).get_material_type_by_one_hot_code(tensor[2:2+wimWarehouse.get_size()].astype(int)))
        # self._window = Window().tensor_to_window(
        #     tensor[2+wimWarehouse.get_size():])

        self.calulate_all()

        #ml_logger.info(
        #    f"tensor {tensor} _material area {self._material.get_area()} _window area {self._window.get_area()}")
        # return Wall(area, orientation, material, window)
        return self

    def add_empty_wall():  # [0,0] + 空材料，空窗户，空窗框

        area_tensor = tf.cast(tf.constant([0, 0]), dtype=tf.float64)
        material_tensor = tf.cast(WallInsulationMaterialWarehouse.get_empty_tensor(
            WallInsulationMaterialWarehouse()), dtype=tf.float64)
        window_tensor = tf.cast(Window.get_empty_tensor(), dtype=tf.float64)

        tensor = tf.concat(
            [area_tensor, material_tensor, window_tensor], axis=0)

        # print("empty wall tensor: ", tensor,"shape: ", tensor.shape)

        return tensor

    # @staticmethod

    def get_wall_features_len(self):
        self.wall_features_len = 2 + wimWarehouse.get_size() + 2 + gmWarehouse.get_size() + \
            wfmWarehouse.get_size()
        return self.wall_features_len

    # 定义一个异常类，用于处理墙数据错误

    class WallDataError(Exception):
        def __init__(self, ErrorInfo):
            super().__init__(self)  # 初始化父类
            self.errorinfo = ErrorInfo

        def __str__(self):
            return self.errorinfo


class Room:

    ROOM_NUMBER = 12
    # ROOM_FEATURES_LEN = 0

    def __init__(self, name='room', type=0, area=0.00, length=0.00, width=0.00, height=2.95,
                 walls=([None]) * 4, floor=None, doors=None, ceiling=None):
        # room_insulation为房间保温材料,door为房间门,window为房间窗户,ceiling为房间天花板,floor为房间地板
        # walls = [0,None] * 4, 0为内墙1为外墙，None为墙体，数组小标为墙体朝向，0为东墙，1为南墙，2为西墙，3为北墙
        self._name = name  # 房间名称
        self._type = type  # 房间类型
        self._area = round(area, 4)  # 房间面积
        self._length = round(length, 4)  # 房间长度
        self._width = round(width, 4)  # 房间宽度
        self._height = round(height, 4)  # 房间高度
        self._walls = walls  # 房间墙体
        self._floor = floor  # 房间地板
        self._doors = doors  # 房间门
        self._ceiling = ceiling  # 房间天花板
        # self.ROOM_FEATURES_LEN = self.get_room_features_len()

        self._cost_view = {}

        self._total_wall_area = 0
        self._total_window_area = 0
        self._avg_ww_ratio = 0
        self._avg_k = 0
        self._wf_ratio = 0
        self._total_wall_num = 0
        self._total_window_num = 0
        self._cost = 0

        self.calculated = False # 2023.7.3修改，增加了一个标志位，用于标志房间是否已经计算过所有参数

    def get_name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    def get_type(self):
        return self._type

    def set_type(self, type):
        self._type = type

    def get_area(self):  # 2023.7.3修改，增加了对房间面积的计算
        self._area = round(self._length * self._width, 4)
        return self._area

    def set_area(self, area):
        self._area = round(area, 4)

    def generate_room_wh(self, room_area):
        ratio = np.random.uniform(0.618, 1.618)  # 生成长宽比，范围为0.618-1.618，符合黄金分割
        length = (room_area * ratio)**0.5
        width = room_area / length
        self.set_length(length)
        self.set_width(width)

    def get_length(self):
        return self._length

    def set_length(self, length):
        self._length = round(length, 4)

    def get_width(self):
        return self._width

    def set_width(self, width):
        self._width = round(width, 4)

    def get_height(self):
        return self._height

    def set_height(self, height):
        self._height = height

    def get_walls(self):
        return self._walls

    def set_walls(self, walls):
        self._walls = walls

    def add_wall(self, orientation, wall):
        if self.check_wall_(orientation, wall):
            self._walls[orientation] = wall

    def remove_wall(self, orientation, wall):
        # if self.check_wall_(orientation,wall):
        self._walls[orientation] = None

    def update_wall(self, wall, orientation):
        self._walls[orientation] = wall

    # 判断墙数据是否合乎房间数据
    def check_wall_(self, orientation, wall):

        # print("wall width = ",wall.get_width(),"room width = ",self.get_length())

        if orientation == Orientation.EAST or orientation == Orientation.WEST:  # 0为东墙，2为西墙

            if wall.get_width() - 1e-3 > self.get_width():  # 判断墙的长度是否合法
                # if wall.get_width() > self.get_width():# 判断墙的长度是否合法
                raise self.RoomDataError("wall width is invalid, wall width is %f, room width is %f" % (
                    wall.get_width(), self.get_width()))

        if orientation == Orientation.SOUTH or orientation == Orientation.NORTH:  # 1为南墙，3为北墙

            if wall.get_width() - 1e-3 > self.get_length():  # 判断墙的长度是否合法
                # if wall.get_width() > self.get_length():# 判断墙的长度是否合法
                raise self.RoomDataError("wall width is invalid, wall width is %f, room width is %f" % (
                    wall.get_width(), self.get_length()))

        if wall.get_height() - 1e-3 > self.get_height():  # 判断墙高度是否合法
            # if wall.get_height() > self.get_height():# 判断墙高度是否合法

            raise self.RoomDataError("wall height is invalid, wall height is %f, room height is %f" % (
                wall.get_height(), self.get_height()))

        return True

    # 判断房间是否符合规范
    def valid_wall_windows_ratio(self, min_ratio, max_ratio):
        total_wall_area = self.get_total_wall_area()
        total_window_area = self.get_total_window_area()
        '''
        print("total_wall_area = ",total_wall_area,"total_window_area = ",
                total_window_area,"ratio = ",total_window_area/total_wall_area,
                "min_ratio = ",min_ratio,"max_ratio = ",max_ratio)
        '''

        if total_wall_area == 0:
            return True

        ratio = total_window_area / total_wall_area

        if ratio > min_ratio and ratio < max_ratio:
            return True
        else:
            return False

    def get_floor(self):
        return self._floor

    def set_floor(self, floor):
        self._floor = floor

    def get_total_wall_area(self):

        if self.calculated == True:
            return self._total_wall_area
    
        self._total_wall_area = 0
        for wall in self._walls:
            if wall is not None and wall._material is not None:
                #self._total_wall_area += wall.calculate_area()
                self._total_wall_area += wall.get_area()

        return round(self._total_wall_area, 4)

    def get_total_window_area(self):

        if self.calculated == True:
            return self._total_window_area
    
        self._total_window_area = 0
        
        for wall in self._walls:
            if wall is not None and wall.get_window() is not None:
                self._total_window_area += wall.get_window().get_area()
        
        return round(self._total_window_area, 4)


    def get_avg_ww_ratio(self):

        if self.calculated == True:
            return self._avg_ww_ratio

        if self.get_total_wall_num() == 0:
            self._avg_ww_ratio = 0
            return self._avg_ww_ratio

        self._total_wall_area = self.get_total_wall_area()
        self._total_window_area = self.get_total_window_area()

        if self._total_wall_area == 0:
            self._avg_ww_ratio = 0
            return self._avg_ww_ratio

        return round(self._total_window_area / self._total_wall_area, 4)

    def get_total_wall_num(self):

        if self.calculated == True:
            return self._total_wall_num

        self._total_wall_num = 0

        for wall in self._walls:
            if wall is not None and wall._material is not None:
                self._total_wall_num += 1
        return self._total_wall_num

    def get_total_window_num(self):

        if self.calculated == True:
            return self._total_window_num

        self._total_window_num = 0
        for wall in self._walls:
            if wall is not None and wall.get_window() is not None:
                self._total_window_num += 1
        return self._total_window_num

    # 定义一个异常类，用于处理房间数据错误
    class RoomDataError(Exception):
        def __init__(self, ErrorInfo):
            super().__init__(self)  # 初始化父类
            self.errorinfo = ErrorInfo

        def __str__(self):
            return self.errorinfo

    # 外墙玻璃，窗框，窗户 总成本
    def get_cost(self):

        if self.calculated == True:
            return self._cost

        self._cost = 0
        for wall in self._walls:
            # if wall is not None and wall[0] == 1 and wall[1] is not None:
            if wall is not None and wall._material is not None:
                self._cost += wall.get_cost()
        return round(self._cost, 2)

    def calculate_all(self):

        self._total_wall_area = 0
        self._total_window_area = 0
        self._avg_ww_ratio = 0
        self._avg_k = 0
        self._wf_ratio = 0
        self._total_wall_num = 0
        self._total_window_num = 0
        self._cost = 0
        
        t_k = 0

        for wall in self._walls :
            if isinstance(wall, Wall) == False:
                continue

            if wall is not None and wall._material is not None:

                self._total_wall_area += wall.calculate_area()
                self._total_window_area += wall.get_window_area()
                self._total_wall_num += 1
                self._total_window_num += 1
                self._cost += wall.get_cost()
                t_k += wall.get_avg_k()*wall.get_area()

        if self._total_wall_area == 0:
            self._avg_ww_ratio = 0
            self._avg_k = 0
        else:
            self._avg_ww_ratio = round(self._total_window_area / self._total_wall_area, 4)
            self._avg_k = round(t_k / self._total_wall_area, 4)
            self._wf_ratio = round(self._total_window_area / self._total_wall_area, 4)

        # ml_logger.info("total_wall_area = %f, total_window_area = %f, avg_ww_ratio = %f, avg_k = %f, wf_ratio = %f, total_wall_num = %d, total_window_num = %d, cost = %f" % (  
        #     self._total_wall_area, self._total_window_area, self._avg_ww_ratio, self._avg_k, self._wf_ratio, self._total_wall_num, self._total_window_num, self._cost))

        self.calculated = True

    def get_cost_view(self):
        self._cost_view = {}

        for wall in self._walls:

            if wall is None:
                continue
            material = wall.get_material()
            # self.add_cost_view(material)
            add_cost_view(self._cost_view, material)
            window = wall.get_window()
            if window is not None:
                # 玻璃
                material = window.get_glass()
                # self.add_cost_view(material)
                add_cost_view(self._cost_view, material)
                # 窗框
                material = window.get_window_frame()
                # self.add_cost_view(material)
                add_cost_view(self._cost_view, material)

        return self._cost_view

    def get_avg_k(self):

        if self.calculated == True:
            return self._avg_k

        self._avg_k = 0
        # print ("self._walls = ",self._walls)
        t_K = 0
        t_a = 0
        for wall in self._walls:

            # if wall is not None and wall[0] == 1 and wall[1] is not None:
            if wall is not None:
                t_K += wall.get_avg_k() * wall.get_area()
                t_a += wall.get_area()

        if t_a == 0:
            self._avg_k = 0
        else:
            self._avg_k = t_K / t_a

        return round(self._avg_k, 4)

    def get_ww_ratio(self):  # 计算墙/窗比值,计算公式为:R=S2/S1；其中S1为墙体面积，S2为窗户面积

        if self.calculated == True:
            return self._avg_ww_ratio

        self._avg_ww_ratio = 0
        t_w = 0
        for wall in self._walls:
            if wall is not None and wall.get_window() is not None:
                t_w += wall.get_window().get_area()

        if self.get_total_wall_area() == 0:
            self._avg_ww_ratio = 0
        else:
            self._avg_ww_ratio = t_w / self.get_total_wall_area()

        return round(self._avg_ww_ratio, 4)

    def get_wf_ratio(self):  # 计算窗/地面面积比

        if self.get_area() > 0:
            return round(self.get_total_window_area()/self.get_area(), 4)
        else:
            return 0

    def get_total_cost(self):  # 计算房间总成本
        self._cost = 0
        for wall in self._walls:
            if wall is not None and wall._material is not None:
                self._cost += wall.get_cost()
        return round(self._cost, 2)

    # 计算外墙材料总面积
    def get_total_material_area(self):
        total_material_area = 0
        for wall in self._walls:
            if wall is not None and wall._material is not None:
                total_material_area += wall.get_material_area()
        return round(total_material_area, 4)

    # 计算外墙材料总成本
    def get_total_material_cost(self):
        material_cost = 0
        for wall in self._walls:
            if wall is not None and wall._material is not None:
                material_cost += wall.get_material_cost()
        return round(material_cost, 4)

    # 计算外墙玻璃总成本

    def get_total_glass_cost(self):
        glass_cost = 0
        for wall in self._walls:
            if wall is not None and wall._material is not None:
                glass_cost += wall.get_glass_cost()
        return round(glass_cost, 4)

    # 计算外墙窗框总成本
    def get_total_frame_cost(self):
        frame_cost = 0
        for wall in self._walls:
            if wall is not None and wall._material is not None:
                frame_cost += wall.get_frame_cost()
        return round(frame_cost, 4)

    # 计算外墙和房屋面积比值
    def get_wr_ratio(self):
        return round(self.get_total_wall_area() / self.get_area(), 4)

    def to_json(self):
        return {
            "room_name": self._name,     # 房间名称
            # _ss.get_room_name(self._type),    # 房间类型
            "room_type": self._type,
            "room_area": self._area,    # 房间面积
            "room_length": self._length,  # 房间长度
            "room_width": self._width,  # 房间宽度
            "room_height": self._height,  # 房间高度
            "total_wall_area": self.get_total_wall_area(),  # 房间外墙面积
            "total_window_area": self.get_total_window_area(),  # 房间窗户面积
            "total_material_area": self.get_total_material_area(),  # 房间外墙材料总面积
            "ww_ratio": self.get_ww_ratio(),  # 房间外墙/窗户比值
            'walls': [wall.to_json() for wall in self._walls if wall is not None and wall._area > 0.0],
            "total_cost_im": self.get_total_material_cost(),  # 房间外墙材料总成本
            "total_cost_g": self.get_total_glass_cost(),  # 房间外墙玻璃总成本
            "total_cost_wf": self.get_total_frame_cost(),  # 房间外墙窗框总成本
            "total_cost": self.get_total_cost(),  # 房间总成本
            "total_avg_k": self.get_avg_k()  # 房间平均导热系数
        }

    # json to room, 用于从json数据中恢复房间对象
    def json_to_room(self, json_data):
        self._type = json_data['room_type']
        self._area = json_data['room_area']
        self._length = json_data['room_length']
        self._width = json_data['room_width']
        self._height = json_data['room_height']
        self._walls = [None]*4

        for wall in json_data['walls']:
            _wall = Wall().json_to_wall(wall)
            self.add_wall(_wall.get_orientation(), _wall)
            # self._walls.append(Wall().json_to_wall(wall))

        # 补充其它空墙
        for i in [Orientation.EAST, Orientation.WEST, Orientation.SOUTH, Orientation.NORTH]:
            if self._walls[i] is None:
                self.add_wall(i, Wall())

        return self

    def to_json_cn(self):
        return {
            "room_name(房间名称)": self._name,     # 房间名称
            # _ss.get_room_name(self._type),    # 房间类型
            "room_type(房间类型)": _ss.get_room_name(self._type),  # self._type,
            "room_area(房间面积)": self._area,    # 房间面积
            "room_length(房间长度)": self._length,  # 房间长度
            "room_width(房间宽度)": self._width,  # 房间宽度
            "room_height(房间高度)": self._height,  # 房间高度
            "total_wall_area(房间外墙面积)": self.get_total_wall_area(),  # 房间外墙面积
            # 房间窗户面积
            "total_window_area(房间窗户面积)": self.get_total_window_area(),
            # 房间外墙材料总面积
            "total_material_area(房间外墙材料总面积)": self.get_total_material_area(),
            "ww_ratio(房间外墙/窗户比值)": self.get_ww_ratio(),  # 房间外墙/窗户比值
            # 'walls': [wall.to_json_cn() for wall in self._walls if wall is not None],
            'walls': [wall.to_json_cn() for wall in self._walls if wall is not None and wall._area > 0.0],
            # 房间外墙材料总成本
            "total_cost_im(房间外墙材料总成本)": self.get_total_material_cost(),
            # 房间外墙玻璃总成本
            "total_cost_g(房间外墙玻璃总成本)": self.get_total_glass_cost(),
            # 房间外墙窗框总成本
            "total_cost_wf(房间外墙窗框总成本)": self.get_total_frame_cost(),
            "total_cost(房间总成本)": self.get_total_cost(),  # 房间总成本
            "total_avg_k(房间平均导热系数)": self.get_avg_k()  # 房间平均导热系数
        }

    # 转换为张量
    # 获取空房间张量 shape=(140,), 140=4+4*34,
    # target = [cost,k] 代表房间cost和 k值

    def to_tensor(self):  # 转换为张量 shape=(140,)


        tensor = tf.cast(tf.constant([self.get_type(),
                                    self.get_area(),
                                    self.get_total_wall_area(),
                                    self.get_total_window_area()]), dtype=tf.float64)
        
        # 把墙拼接到张量中
        for wall in self._walls:
            if wall is not None:
                tensor = tf.concat([tensor, wall.to_tensor()], axis=0)
            else:
                tensor = tf.concat([tensor, Wall.add_empty_wall()], axis=0)


        target = tf.cast(tf.constant(
            [self.get_total_cost(), self.get_avg_k()]), dtype=tf.float64)
        

        # print("room tensor ", tensor,"shape ", tensor.shape, "target ", target)
        # print("room tensor shape ", tensor.shape, "target shape ", target.shape)
        #ml_logger.info(
        #    f"convert room {self._name} to tensor, shape {tensor.shape} target shape {target.shape}")
        return tensor, target

    # tensor to room
    # 用于从张量中恢复房间对象
    def tensor_to_room(self, tensor):

        # pdb.set_trace()
        #ml_logger.info(f"------------------- Tensor to room {self._name}------------------------")
        #ml_logger.info(f"tensor {tensor} shape {tensor.shape}")

        self._type = tensor[0]
        self._area = tensor[1]
        # self._walls = [None]*4
        # f_len = Wall().get_wall_features_len()
        f_len = Wall.WALL_FEATURES_LEN

        for wall_index, wall in enumerate(self._walls):
            if isinstance(wall, Wall):
                t_data = tensor[4+wall_index*f_len:4+(wall_index+1)*f_len]
                if int(t_data[0]) == 0:
                    #ml_logger.warning(
                    #    f"wall {wall_index} is empty, continue.")
                    continue

                wall = wall.tensor_to_wall(t_data)
                #ml_logger.info(
                #    f"update wall {wall_index} tensor_data {t_data} shape {t_data.shape} orientation {wall.get_orientation()}")
                self.update_wall(copy.deepcopy(wall), wall.get_orientation())
                    #wall, wall.get_orientation())
            else:
                ml_logger.warning(
                    f"Element at index {wall_index} is not an instance of Wall.")

        self.calculate_all()

        # for i in range(self.get_total_wall_num()):
        #     self.add_wall(i,
        #                   Wall().tensor_to_wall(
        #                       tensor[2+i*f_len:2+(i+1)*f_len])
        #                   )

        return self

    # 获取空房间张量 shape=(140,), 140=4+4*34,
    # target = [0,0] 代表房间cost和 k值
    @ staticmethod
    def get_empty_room_tensor():
        tensor = tf.cast(tf.constant([0, 0, 0, 0]), dtype=tf.float64)

        for i in range(4):
            tensor = tf.concat([tensor, Wall.add_empty_wall()], axis=0)

        target = tf.cast(tf.constant([0, 0]), dtype=tf.float64)

        return tensor, target

    # @staticmethod
    def get_room_features_len(self):
        return self.get_empty_room_tensor()[0].shape[0]

    ROOM_FEATURES_LEN = 4 + 4 * Wall.WALL_FEATURES_LEN #int(get_empty_room_tensor()[0].shape[0])
    ROOM_TARGETS_LEN = 2
# STAIRCASE


HOUSE_TYPE_STAIRCASE = 2
HOUSE_TYPE_CORRIDOR = 1
HOUSE_TYPE_APARTMENT = 0


# 公共空间,楼梯间，走廊

class PublicSpace(Room):
    _type = HOUSE_TYPE_STAIRCASE

    def __init__(self, space_type, **kwargs):
        super().__init__(**kwargs)
        self._type = space_type

    def get_type(self):
        return self._type


def add_cost_view(cost_view, material):
    # pdb.set_trace()
    if material is None:
        return
    if isinstance(material, Glass):
        material_type = material.get_material_type()['descriptions']
    elif isinstance(material, WallInsulation):
        material_type = material.get_material_type()['name']
    elif isinstance(material, WindowFrame):
        material_type = material.get_material_type()['type']
    else:
        material_type = material.get_material_type()

    if any(material_type == key for key in cost_view):

        cost_view[material_type]['area'] = round(
            cost_view[material_type]['area'] + material.get_area(), 4)
        cost_view[material_type]['cost'] = round(
            cost_view[material_type]['cost'] + material.get_cost(), 2)


class House:

    # 楼梯间，
    # HOUSE_TYPE_STAIRCASE = 2
    # 连廊,走廊
    # HOUSE_TYPE_CORRIDOR = 1
    # 公寓
    # HOUSE_TYPE_APARTMENT = 0

    wall_features_len = Wall().get_wall_features_len()
    room_features_len = Room().get_room_features_len()
    # ml_logger.info(
    #    f"wall_features_len {wall_features_len} room_features_len {room_features_len}")
    # HOUSE_FEATURES = 1680
    # TARGET = 26

    HOUSE_FEATURES =int(_ss.get_max_num_rooms() * room_features_len)
    TARGET = int(_ss.get_max_num_rooms() * 2 + 2)

    # ml_logger.info(f"HOUSE_FEATURES {HOUSE_FEATURES} TARGET {TARGET}")

    def __init__(self, type=0, name='House', area=0.0, height=2.95, room_types=[], rooms=[]):
        self._type = type
        self._name = name
        self._area = area
        self._height = height
        self._room_types = room_types
        self._rooms = rooms
        self._cost_view = {}

        self.calculated = False

    def get_name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    def get_area(self):
        return self._area

    def set_area(self, area):
        self._area = area

    def get_height(self):
        return self._height

    def set_height(self, height):
        self._height = height

    def get_room_types(self):
        return self._room_types


    def get_rooms(self):
        return self._rooms

    def set_rooms(self, rooms):
        self._rooms = rooms

    def add_room(self, room):
        self._rooms.append(room)

    def update_room(self, room, index):
        # index = self._rooms.index(target)
        if index >= 0:
            self._rooms[index] = room
        else:
            ml_logger.warning(f"update_room {index} not in {self._rooms}")

    def del_room(self, room):
        self._rooms.remove(room)

    def get_room_num(self):
        return len(self._rooms)

    def get_total_cost(self):

        if self.calculated:
            return round(self._cost,2)

        self._cost = 0
        for room in self._rooms:
            self._cost += room.get_cost()

        self._cost = round(self._cost, 2)
        return self._cost#round(self._cost, 2)

    # def get_total_area(self):  # 计算房屋面积

    #     self._area = 0
    #     for room in self._rooms:
    #         self._area += room.get_area()
    #     return round(self._area, 4)
        # self._total_area = 0
        # for room in self._rooms:
        #     self._total_area += room.get_area()
        # return round(self._total_area, 4)

    def get_total_window_area(self):  # 计算房屋窗户面积
 
        if self.calculated:
            return round(self._total_window_area,4)

        self._total_window_area = 0
        for room in self._rooms:
            self._total_window_area += room.get_total_window_area()

        self._total_window_area = round(self._total_window_area, 4)

        return self._total_window_area#round(self._total_window_area, 4)

    def get_total_wall_num(self):  # 计算房屋外墙数量
  
        if self.calculated:
            return self._total_wall_num

        self._total_wall_num = 0
        for room in self._rooms:
            self._total_wall_num += room.get_total_wall_num()
        
        self._total_wall_num = round(self._total_wall_num, 4)

        return self._total_wall_num

    def get_total_window_num(self):  # 计算房屋窗户数量

        if self.calculated:
            return round(self._total_window_num,4)

        self._total_window_num = 0
        for room in self._rooms:
            self._total_window_num += room.get_total_window_num()
        self._total_window_num = round(self._total_window_num, 4)
        return self._total_window_num

    def get_total_wall_area(self):  # 计算房屋外墙面积

        if self.calculated:
            return round(self._total_wall_area,4)

        self._total_wall_area = 0
        for room in self._rooms:
            self._total_wall_area += room.get_total_wall_area()
        
        self._total_wall_area = round(self._total_wall_area, 4)
        return self._total_wall_area#round(self._total_wall_area, 4)
    # 外墙材料面积

    def get_total_material_area(self):

        if self.calculated:
            return round(self._total_material_area,4)

        self._total_material_area = 0
        for room in self._rooms:
            self._total_material_area += room.get_total_material_area()
        
        self._total_material_area = round(self._total_material_area, 4)
        return self._total_material_area#round(self._total_material_area, 4)

    def get_avg_ww_ratio(self):  # 计算平均墙/窗比值

        if self.calculated:
            return round(self._avg_ww_ratio,4)

        if self.get_total_area() == 0:
            ml_logger.warning(f"House {self.get_name()} get_total_area() == 0")
            return 0

        self._avg_ww_ratio = 0
        for room in self._rooms:
            self._avg_ww_ratio += room.get_ww_ratio() * room.get_area()
        self._avg_ww_ratio = round(self._avg_ww_ratio / self.get_total_area(),4)
        return self._avg_ww_ratio#round(self._avg_ww_ratio, 4)

    def get_avg_k(self):  # 计算平均传热系数

        if self.calculated:
            return round(self._avg_k,4)

        if self.get_total_area() == 0:
            ml_logger.warning(f"House {self.get_name()} get_total_area() == 0")
            return 0

        self._avg_k = 0

        for room in self._rooms:
            self._avg_k += room.get_avg_k() * room.get_area()

        self._avg_k = round(self._avg_k / self.get_total_area(),4)

        return self._avg_k #round(self._avg_k, 4)

    def get_total_area(self):  # 计算房屋总面积

        if self.calculated:
            return round(self._area,4)

        self._area = 0
        for room in self._rooms:
            self._area += room.get_area()
        return round(self._area, 4)

    # get cost view,汇总不同型号材料所用的面积、数量、价格、总价
    # 包括 glass,window_frame,wall,wall_insulation
    # return {'material':{'area':,'price':,'cost':}}

    # 在创建house后，计算所有材料的面积、数量、价格、总价
    def calculate_all(self):


        self._area = 0.0
        self._total_window_area = 0.0
        self._total_wall_num = 0
        self._total_window_num = 0
        self._total_wall_area = 0.0
        self._total_material_area = 0.0
        self._avg_ww_ratio = 0.0
        self._avg_k = 0.0
        self._cost = 0.0

        t_k = 0.0

        for room in self._rooms:

            if isinstance(room, Room):
                
                room.calculate_all()
                
                self._area += room.get_area()
                self._total_window_area += room.get_total_window_area()
                self._total_wall_num += room.get_total_wall_num()
                self._total_window_num += room.get_total_window_num()
                self._total_wall_area += room.get_total_wall_area()
                self._total_material_area += room.get_total_material_area()
                self._avg_ww_ratio += room.get_ww_ratio() * room.get_area()
                t_k += room.get_avg_k() * room.get_area()
                self._cost += room.get_cost()

        if self._area == 0:
            ml_logger.warning(f"House {self.get_name()} get_total_area() == 0")
            self._avg_ww_ratio = 0.0
            self._avg_k = 0.0

        else:
            self._avg_k = round(t_k / self._area, 4)

        self.calculated = True


    def get_cost_view(self):
        # pdb.set_trace()
        self._cost_view = {}
        for room in self._rooms:
            for wall in room.get_walls():
                # 外墙
                if wall is None:
                    continue
                material = wall.get_material()
                self.add_cost_view(material)
                # add_cost_view(self._cost_view, material)
                window = wall.get_window()
                if window is not None:
                    # 玻璃
                    material = window.get_glass()
                    self.add_cost_view(material)
                    # add_cost_view(self._cost_view, material)
                    # 窗框
                    material = window.get_window_frame()
                    self.add_cost_view(material)
                    # add_cost_view(self._cost_view, material)
        return self._cost_view

    # add material to cost_view

    def add_cost_view(self, material):
        # pdb.set_trace()
        if material is None:
            return
        if isinstance(material, Glass):
            material_type = material.get_material_type()['descriptions']
        elif isinstance(material, WallInsulation):
            material_type = material.get_material_type()['name']
        elif isinstance(material, WindowFrame):
            material_type = material.get_material_type()['type']
        else:
            material_type = material.get_material_type()

        if any(material_type == key for key in self._cost_view):

            self._cost_view[material_type]['area'] = round(
                self._cost_view[material_type]['area'] + material.get_area(), 4)
            self._cost_view[material_type]['cost'] = round(
                self._cost_view[material_type]['cost'] + material.get_cost(), 2)

        else:
            # Convert the dictionary to a string representation
            key = str(material_type)
            self._cost_view[key] = {
                'area': round(material.get_area(), 4),
                'price': material.get_price(),
                'cost': round(material.get_cost(), 2)
            }

    def to_json(self):
        return {
            'h_name': self._name,  # 房屋名称
            'h_total_area': self.get_total_area(),  # 房屋总面积
            'h_total_room_num': self.get_room_num(),  # 房屋房间数量
            'h_total_wall_num': self.get_total_wall_num(),  # 房屋外墙数量
            'h_total_wall_area': self.get_total_wall_area(),  # 房屋外墙面积
            'h_total_window_area': self.get_total_window_area(),  # 房屋窗户面积
            'h_total_material_area': self.get_total_material_area(),  # 房屋外墙材料面积
            'h_total_window_num': self.get_total_window_num(),  # 房屋窗户数量
            'h_height': self._height,  # 房屋高度
            'h_total_avg_k': self.get_avg_k(),  # 房屋平均导热系数
            'h_total_avg_ww_ratio': self.get_avg_ww_ratio(),  # 房屋平均墙/窗比值
            'h_total_cost': self.get_total_cost(),  # 房屋总材料成本
            'rooms': [room.to_json() for room in self._rooms]
        }

    def json_to_house(self, json_data):
        # house = House()
        self.set_name(json_data['h_name'])
        self.set_height(json_data['h_height'])
        rooms = []
        for room in json_data['rooms']:
            rooms.append(Room().json_to_room(room))
        self.set_rooms(rooms)
        # self.set_rooms([Room.json_to_room(room) for room in json['rooms']])
        return self

    def to_json_cn(self):
        return {
            'h_name [房屋名称]': self._name,  # 房屋名称
            'h_total_area [房屋总面积]': self.get_total_area(),  # 房屋总面积
            'h_total_room_num [房屋房间数量]': self.get_room_num(),  # 房屋房间数量
            'h_total_wall_num [房屋外墙数量]': self.get_total_wall_num(),  # 房屋外墙数量
            'h_total_wall_area [房屋外墙面积]': self.get_total_wall_area(),  # 房屋外墙面积
            # 房屋窗户面积
            'h_total_window_area [房屋窗户面积]': self.get_total_window_area(),
            # 房屋外墙材料面积
            'h_total_material_area [房屋外墙材料面积]': self.get_total_material_area(),
            # 房屋窗户数量
            'h_total_window_num [房屋窗户数量]': self.get_total_window_num(),
            'h_height [房屋高度]': self._height,  # 房屋高度
            'h_total_avg_k [房屋平均导热系数]': self.get_avg_k(),  # 房屋平均导热系数
            # 房屋平均墙/窗比值
            'h_total_avg_ww_ratio [房屋平均墙/窗比值]': self.get_avg_ww_ratio(),
            'h_total_cost [房屋总材料成本]': self.get_total_cost(),  # 房屋总材料成本
            'rooms': [room.to_json_cn() for room in self._rooms]
        }

    class CustomJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            return super().default(obj)

    def save_to_json(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(self.to_json(), json_file, ensure_ascii=False,
                      indent = 4, cls = self.CustomJSONEncoder)

    def save_to_json_cn(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(self.to_json_cn(), json_file, ensure_ascii=False,
                      indent = 4, cls = self.CustomJSONEncoder)

    # room_tensors's shape = (1680,) = ROOM_NUMBER* room_tensor's shape(140,) + 4,
    # target's shape ROOM_NUMBER*2 + 2 = 26

    def to_tensor(self):
        # pdb.set_trace()
        room_tensors = []
        targets = []

        for room in self._rooms:
            if room is not None:
                r_tensor, r_target = room.to_tensor()
                room_tensors.append(r_tensor)
                targets.append(r_target)
        # 补满空房间
        for i in range(Room.ROOM_NUMBER - len(room_tensors)):
            empty_tensors, empty_targets = Room.get_empty_room_tensor()

            room_tensors.append(empty_tensors)
            targets.append(empty_targets)

        targets.append([self.get_total_cost(), self.get_avg_k()])

        room_tensors = tf.concat(room_tensors, axis=0)
        targets = tf.concat(targets, axis=0)
        targets = tf.cast(targets, tf.float64)

        #ml_logger.info(
        #    f"convert room {self._name} to tensor  {room_tensors.shape} targets.shape {targets.shape}")

        return room_tensors, targets

    @ staticmethod
    def get_empty_house_tensor():

        room_tensors= []
        targets= []
        for i in range(Room.ROOM_NUMBER):
            empty_tensors, empty_targets= Room.get_empty_room_tensor()
            room_tensors.append(empty_tensors)
            targets.append(empty_targets)
        targets.append([0, 0])  # total_cost,avg_k

        room_tensors= tf.concat(room_tensors, axis=0)
        targets= tf.concat(targets, axis=0)
        targets= tf.cast(targets, tf.float64)

        return room_tensors, targets

    # tensor to house
    def tensor_to_house(self, house_tensor):  # , targets):

        # rooms = []
        #ml_logger.info(f"convert tensor to house {self._name} {house_tensor.shape} rooms {self.get_room_num()}")

        for i in range(self.get_room_num()):
            room= self._rooms[i]

            if isinstance(room, Room):
                room_tensor = house_tensor[i*room.ROOM_FEATURES_LEN:(i+1)*room.ROOM_FEATURES_LEN]
                room= room.tensor_to_room(room_tensor)
                #room.calculate_all()
                # targets[i*2:(i+1)*2])
                # targets[i*Room.ROOM_TARGETS_LEN:(i+1)*Room.ROOM_TARGETS_LEN])
                self.update_room(copy.deepcopy(room), i)
        self.calculate_all()

        # self.set_total_cost(targets[-2])
        # self.set_avg_k(targets[-1])
        return self

    # wall_features_len = Wall().get_wall_features_len()
    # room_features_len = Room().get_room_features_len()
    wall_features_len = Wall.WALL_FEATURES_LEN
    room_features_len = Room.ROOM_FEATURES_LEN
    # 根据tensor 中的index 找到对应的feature name
    
    def find_feature_name(self, index):

        point = index % self.room_features_len  # Room.ROOM_NUMBER
        # ml_logger.info(f"--------------index {index} ,point {point} ")
        if point < 4:
            # print("point <4 ", point )
            return "No mutation allowed"

        # if point >= 4  and  point < 4 + wimWarehouse.get_size():
        #    return "wall_material"
        else:
            bpoint = point
            point = point - 4

            point = point % self.wall_features_len  # Wall().get_wall_features_len()

            # ml_logger.info(f"point wall-disct index-bpoint-point,{self.wall_features_len}-{index}-{bpoint}-{point}")

            # ml_logger.info(f"############# index {index} ,point {point} ")

            wim_size = wimWarehouse.get_size()
            gm_size = gmWarehouse.get_size()


            if point >= 2 and point < 2 + wimWarehouse.get_size():
                return "wall_material"
            # and point < 2 + wimWarehouse.get_size() + 2:
            elif point == 2 + wim_size:
                return "window_size"
            elif point >= 2 + wim_size + 2 and point < 2 + wim_size + 2 + gm_size:
                return "glass_material"
            elif point >= 2 + wim_size + 2 + gm_size:
                return "wf_material"
            else:
                return "No mutation allowed"

        return "No mutation allowed"


# 一个创建房屋的类
class HousesCreator:

    def __init__(self, number):
        self._houses = []
        self.orientation = Orientation()
        self._number = number

    def make_houses(self):
        for i in range(self._number):
            self._houses.append(self.make_house(name=f"House {i+1}"))
        return self._houses

    def make_house(self, name="My House"):

        # def calculate_window_dimensions(window_area, wall_width, wall_height):
        #     aspect_ratio = np.random.uniform(0.618, 1.618)
        #     window_width = np.sqrt(window_area * aspect_ratio)
        #     window_height = window_area / window_width

        #     if window_width > wall_width:
        #         window_width = wall_width
        #         window_height = window_area / window_width

        #     if window_height > wall_height:
        #         window_height = wall_height
        #         window_width = window_area / window_height

        #     return window_width, window_height

        # 创建一个房屋对象,房屋的高度为标准层高
        house = House(name=name, height=_ss.get_standard_floor_height(),
                      room_types=_ss.get_random_rooms(),rooms=[])

        ml_logger.info(f"house._room_types {house._room_types}")

        for index in house._room_types: # index 是房间的类型,rooms 为房间的列表
        
            ml_logger.info(f"index of house._rooms {index}")

            (min_area, max_area, min_wall_num, max_wall_num,
             required, min_room_num, max_room_num, min_window_num, max_window_num, wo_probability) = _ss.get_room_encoding_info(index)
            
            ml_logger.info(f"min_area, max_area, min_wall_num, max_wall_num, required, min_room_num, max_room_num, min_window_num, max_window_num, wo_probability {min_area, max_area, min_wall_num, max_wall_num, required, min_room_num, max_room_num, min_window_num, max_window_num, wo_probability}")
            
            room_area = np.random.uniform(min_area, max_area)  # 生成一个房间面积
            min_window_wall_ratio, max_window_wall_ratio = _ss.get_room_wwr_range(
                index)  # 5.19 改为独立获取每个房间的窗墙比
            min_window_floor_ratio, max_window_floor_ratio = _ss.get_room_wfr_range(
                index)  # 7.24 增加 窗地面积比校验

            # 使用优化器 预生成房间的优化好的窗户数据，房间长宽数据 2023.7.26
            window_opt = WindowOptimization(
                room_area, house.get_height(),
                [0.35, 0.45, 0.35, 0.40],  # 各个方向的窗墙比，来自规范
                [1/6, (1 / 6)*1.4]  # 窗地比
            )
            w_areas = window_opt.optimize()  # 窗户面积元组
            LWH = window_opt.get_LWH()  # 长宽高元组
            # print(w_areas, LWH)

            # 创建一个房间对象,walls=[0,None] * 4表示房间的四面墙都没有窗户
            room = Room(name=f"Room {index} ", type=index,
                        area=room_area, length=LWH[0], width=LWH[1], height=house.get_height(
                        ),
                        walls=[None] * 4, floor=None, doors=[], ceiling=None)

            # 生成一个房间的墙的数量,min_wall_num和max_wall_num是墙的数量的范围
            wall_num = np.random.randint(min_wall_num, max_wall_num + 1)

            if wall_num == 0:  # 如果墙的数量为0,则不需要创建新的外墙
                # house.update_room(room, index)
                continue

            for o_index in range(len(w_areas)):  # o_index 序号和方位代码重合

                if w_areas[o_index] > 0:

                    wall_width = room.get_width() if o_index in [
                        Orientation.EAST, Orientation.WEST] else room.get_length()

                    # 生成一个墙的面积
                    r_height = room.get_height()
                    exterior_wall_area = round(wall_width * r_height, 4)
                    # 生成一个墙对象
                    wall = Wall(area=exterior_wall_area,  # 生成一个墙的面积
                                width=wall_width, height=r_height, thickness=0.0,  # 生成一个墙的宽度,高度,厚度
                                # 生成一个墙的材料,缺省 外墙材料面积为墙的面积，在生成窗后，减除窗的面积
                                wi_material=WallInsulation(
                                    area=exterior_wall_area, warehouse=wimWarehouse, material_type=wimWarehouse.get_random_material()),
                                orientation=o_index,  # 生成一个墙的朝向
                                window=None  # 生成一个墙的窗户
                                )

                    window_area = w_areas[o_index]

                    # 根据窗户的面积和宽高比约束来计算窗户的宽度和高度
                    # 如果窗户的宽度或高度超过了墙的宽度或高度，则按比例缩小窗户的尺寸,
                    # 保证窗户不超过墙的尺寸,保证窗户的宽高比在0.618~1.618之间

                    window_width, window_height = calculate_window_dimensions(
                        window_area, wall.get_width(), wall.get_height())

                    # 生成一个窗户对象
                    window = Window(area=window_area, width=window_width,
                                    height=window_height, orientation=wall.get_orientation())
                    # 生成一个窗框对象
                    # 生成一个窗框的材料,并将窗框的材料设置到窗框上,并计算窗框的面积比例
                    wfa_ratio, wf_material = wfmWarehouse.get_best_wf(window_width, window_height)
                    # 将窗框的材料设置到窗框上
                    window_frame = WindowFrame(
                        area=window_area*wfa_ratio, warehouse=wfmWarehouse, material_type=wf_material)
                    # 生成一个窗框对象，将窗框添加窗户到上
                    window.set_window_frame(window_frame)
                    # 生成一个玻璃对象，将玻璃添加到窗户上
                    glass = Glass(area=window_area*(1-wfa_ratio), warehouse=gmWarehouse,
                                  material_type=gmWarehouse.get_random_material())
                    window.set_glass(glass)
                    # 将窗户添加到墙上
                    wall.add_window(window)
                    # 把材料，窗户，框的面积set 完整
                    #
                    # 将墙添加到房间中
                    room.add_wall(o_index, wall)

            # 将房间添加到房屋中
            house.add_room(room)

        house.calculate_all()

        return house

def calculate_window_dimensions(window_area, wall_width, wall_height):
    aspect_ratio = np.random.uniform(0.618, 1.618)
    window_width = np.sqrt(window_area * aspect_ratio)
    window_height = window_area / window_width

    if window_width > wall_width:
        window_width = wall_width
        window_height = window_area / window_width

    if window_height > wall_height:
        window_height = wall_height
        window_width = window_area / window_height

    return window_width, window_height
class publicSpaceCreator(object):
    """docstring for StaircaseCreator"""
    space_code = 12
    space_height = 2.95

    def __init__(self, number, space_code=12):
        self.number = number
        self.public_spaces = []
        self.space_code = space_code

    def make_public_spaces(self):

        for index in range(self.number):
            public_space = self.make_public_space()
            self.public_spaces.append(public_space)

        return self.public_spaces

    def make_public_space(self):
        # def calculate_window_dimensions(window_area, wall_width, wall_height):
        #     aspect_ratio = np.random.uniform(0.618, 1.618)
        #     window_width = np.sqrt(window_area * aspect_ratio)
        #     window_height = window_area / window_width

        #     if window_width > wall_width:
        #         window_width = wall_width
        #         window_height = window_area / window_width

        #     if window_height > wall_height:
        #         window_height = wall_height
        #         window_width = window_area / window_height

        #     return window_width, window_height

        (min_area, max_area, min_wall_num, max_wall_num,
            required, min_room_num, max_room_num, min_window_num, max_window_num, wo_probability) = _ss.get_room_encoding_info(self.space_code)

        space_area = np.random.uniform(min_area, max_area)  # 生成一个房间面积
        min_window_wall_ratio, max_window_wall_ratio = _ss.get_room_wwr_range(
            self.space_code)  # 12 为 staircase 的room 类型代码
        min_window_floor_ratio, max_window_floor_ratio = _ss.get_room_wfr_range(
            self.space_code)  # 7.24 增加 窗地面积比校验

        # 使用优化器 预生成房间的优化好的窗户数据，房间长宽数据 2023.7.26
        window_opt = WindowOptimization(
            space_area, self.space_height,
            [0.35, 0.45, 0.35, 0.40],  # 各个方向的窗墙比，来自规范
            [1/6, (1 / 6)*1.4]  # 窗地比
        )
        w_areas = window_opt.optimize()  # 窗户面积元组
        LWH = window_opt.get_LWH()  # 长宽高元组
        # print(w_areas, LWH)

        # 创建一个房间对象,walls=[0,None] * 4表示房间的四面墙都没有窗户
        public_space = PublicSpace(space_type=self.space_code, name=f"public_space",
                                   area=space_area, length=LWH[0], width=LWH[1], height=self.space_height,
                                   walls=[None] * 4, floor=None, doors=[], ceiling=None)

        # 生成墙的数量,min_wall_num和max_wall_num是墙的数量的范围
        wall_num = np.random.randint(min_wall_num, max_wall_num + 1)

        if wall_num == 0:  # 如果墙的数量为0,则不需要创建新的外墙
            return public_space

        for o_index in range(len(w_areas)):  # o_index 序号和方位代码重合

            if w_areas[o_index] > 0:

                wall_width = public_space.get_width() if o_index in [
                    Orientation.EAST, Orientation.WEST] else public_space.get_length()

                # 生成一个墙的面积
                r_height = public_space.get_height()
                exterior_wall_area = round(wall_width * r_height, 4)
                # 生成一个墙对象
                wall = Wall(area=exterior_wall_area,  # 生成一个墙的面积
                            width=wall_width, height=r_height, thickness=0.0,  # 生成一个墙的宽度,高度,厚度
                            # 生成一个墙的材料,缺省 外墙材料面积为墙的面积，在生成窗后，减除窗的面积
                            wi_material=WallInsulation(
                                area=exterior_wall_area, warehouse=wimWarehouse, material_type=wimWarehouse.get_random_material()),
                            orientation=o_index,  # 生成一个墙的朝向
                            window=None  # 生成一个墙的窗户
                            )

                window_area = w_areas[o_index]

                # 根据窗户的面积和宽高比约束来计算窗户的宽度和高度
                # 如果窗户的宽度或高度超过了墙的宽度或高度，则按比例缩小窗户的尺寸,
                # 保证窗户不超过墙的尺寸,保证窗户的宽高比在0.618~1.618之间

                window_width, window_height = calculate_window_dimensions(
                    window_area, wall.get_width(), wall.get_height())

                # 生成一个窗户对象
                window = Window(area=window_area, width=window_width,
                                height=window_height, orientation=wall.get_orientation())
                # 生成一个窗框对象
                # 生成一个窗框的材料,并将窗框的材料设置到窗框上,并计算窗框的面积比例
                wfa_ratio, wf_material = wfmWarehouse.get_best_wf(
                    window_width, window_height)
                # 将窗框的材料设置到窗框上
                window_frame = WindowFrame(
                    area=window_area*wfa_ratio, warehouse=wfmWarehouse, material_type=wf_material)
                # 生成一个窗框对象，将窗框添加窗户到上
                window.set_window_frame(window_frame)
                # 生成一个玻璃对象，将玻璃添加到窗户上
                glass = Glass(area=window_area*(1-wfa_ratio), warehouse=gmWarehouse,
                              material_type=gmWarehouse.get_random_material())
                window.set_glass(glass)
                # 将窗户添加到墙上
                wall.add_window(window)
                # 把材料，窗户，框的面积set 完整
                #

                # 将墙添加到房间中
                public_space.add_wall(o_index, wall)

        return public_space


def main():

    # configurator = GPUConfigurator(use_gpu=True, gpu_memory_limit=2048)
    # configurator.configure_gpu()
    # tf.device(configurator.select_device())

    hc = HousesCreator(10)
    # 生成一个房屋对象
    houses = hc.make_houses()
    for house in houses:
        house.save_to_json(f"./houses_json/house_{house._name}.json")
        house.save_to_json_cn(f"./houses_json/cn_house_{house._name}.json")

        # print("-------------------------------------------------------------------")
        # print("house tensor", house.to_tensor())
        houses_features, target = house.to_tensor()
        # print("houses_features ", houses_features,
        #       "shape *****************:", houses_features.shape)
        # print("target ", target, "shape *****************:", target.shape)
        # print("*******************************************************************")
        # print(house.get_cost_view())


def load_json_to_house(json_file_path):
    json_data = None
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
    house = House()
    house.json_to_house(json_data)

    house.save_to_json(f"{json_file_path}_update.json")


if __name__ == "__main__":
    main()
    # load_json_to_house("./houses_json/training_dataset_json/20230611-135209-325994")
