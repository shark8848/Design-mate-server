import os
current_path = os.getcwd()
print('current_path',current_path)

import numpy as np
import random
import json
import os
import tensorflow as tf
import pdb
import sys
#sys.path.append("..")
import sys
import os

# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将 ml_server 目录路径添加到模块搜索路径中
sys.path.append(current_dir)

from ml_server.SpaceStandard import SpaceStandard
from ml_server.MaterialWarehouse import GlassMaterialWarehouse,WallInsulationMaterialWarehouse,WindowFrameMaterialWarehouse
from apocolib.MlLogger import mlLogger as ml_logger

gmWarehouse = GlassMaterialWarehouse()
wimWarehouse = WallInsulationMaterialWarehouse()
wfmWarehouse = WindowFrameMaterialWarehouse()
_ss = SpaceStandard() #创建空间标准对象

# 定义材料基础类
class Material:
    def __init__(self, area=0.0, material_type=None, warehouse = None, price=0.0, K=0.0):
        self._area = round(area,4)
        self._material_type = material_type
        self.warehouse = warehouse
        self.price = round(price,2)
        self.K = round(K,4)

    def get_area(self):
        return self._area

    def set_area(self, area):
        self._area = round(area,4)

    def get_material_type(self):
        return self._material_type

    def set_material_type(self, material_type):
        self._material_type = material_type

    def set_warehouse(self, warehouse):
        self.warehouse = warehouse
    
    def get_warehouse(self):
        return self.warehouse

    def get_cost(self):
        #print('material_type:',self._material_type,'price:',self.price,'area:',self._area)
        return round(self._area * self.price, 4)

    def get_K(self):
        return self.K

    def get_price(self):
        return self.price

    def set_price(self, price):
        self.price = round(price,2)

    # 生成独热编码
    def get_one_hot_code(self):
        encoding = [0] * self.warehouse.get_size()
        #print('encoding:',encoding)
        #print('material_type:',self._material_type)
        #print('key = ',self.warehouse.find_key(self._material_type))

        if self._material_type:
            key = self.warehouse.find_key(self._material_type)
            if key is not None:
                encoding[int(key)] = 1
        #print('encoding:',encoding)
        # to tensor
        encoding = tf.convert_to_tensor(encoding, dtype=tf.float64)
        
        return encoding

    # 根据独热编码生成材料类型
    def get_material_type_by_one_hot_code(self, encoding):
        key = np.argmax(encoding)
        print("encoding ",encoding, " key ",key)
        #return key
        return self.warehouse.get_material(key)

class Glass(Material):
    def __init__(self, area=0.0, width=0.0, height=0.0, material_type=None, warehouse = None, price=0.0,K=0.0):

        if material_type is not None:
            price = material_type['price']
            K = material_type['K']
        
        Material.__init__(self,area, material_type,warehouse,price,K)
        self._width = round(width,4)
        self._height = round(height,4)

    def get_width(self):
        return self._width

    def get_height(self):
        return self._height

    def get_cost(self):
        return super().get_cost()
    

class WindowFrame(Material):
    def __init__(self, area=0.0, length = 0.0, material_type=None, warehouse = None, price=0.0,K=0.0):

        if material_type is not None:
            price = material_type['price']
            K = material_type['K']
        
        super().__init__(area, material_type,warehouse,price,K)
        self._length = length

    def get_length(self):
        return self._length

    def set_length(self, length):
        self._length = round(length,4)
    
    def get_cost(self):
        return super().get_cost()


class WallInsulation(Material):
    def __init__(self, area=0, thickness =0.0,material_type=None, warehouse = None,price=0.0,K=0.0):

        if material_type is not None:
            price = material_type['price']
            K = material_type['K']

        super().__init__(area, material_type,warehouse,price,K)
        self._thickness = round(thickness,2)
    
    def get_thickness(self):
        return self._thickness

    def set_thickness(self, thickness):
        self._thickness = round(thickness,2)

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

    def get_orientation_OHE(self,orientation) -> np.ndarray:
        one_hot_encoding = np.zeros(len(self.orientation_mapping))
        one_hot_encoding[orientation] = 1
        return one_hot_encoding

    def get_orientation_OHL(self) -> np.ndarray:
        one_hot_list = []
        for i in range(len(self.orientation_mapping)):
            one_hot_list.append(self.get_orientation_OHE(i))
        return one_hot_list

    #print(get_orientation_OHL())
    # 查询所有朝向的key
    #@staticmethod
    def get_orientation_keys(self):
        return list(self.orientation_mapping.keys())

    # 查询朝向数量
    #@staticmethod
    def get_o_num(self) -> int:
        return len(self.orientation_mapping)

    # 查询朝向
    #@staticmethod
    def get_orientation(self,orientation):
        for key, value in self.orientation_mapping.items():
            if value == orientation:
                return key
        return None

    def get_orientation(self, orientation):
        return self.orientation_mapping[orientation]

    def get_orientation_mapping(self):
        return self.orientation_mapping


class Window:
    def __init__(self, area=0.0, width=0.0, height=0.0, glass=None, window_frame=None,orientation=None):
        self._area = round(area,4)
        self._width = round(width,4)
        self._height = round(height,4)
        self._glass = glass
        self._window_frame = window_frame
        self._orientation = orientation

    def get_area(self):
        self._area = round(self._width * self._height,4)
        return self._area

    def get_width(self):
        return self._width

    def set_width(self, width):
        self._width = round(width,4)

    def get_height(self):
        return self._height

    def set_height(self, height):
        self._height = round(height,4)

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
        return round(self._glass.get_cost() + self._window_frame.get_cost(),4)

    # 计算窗户的平均K值,单位为W/(m2*K),即窗户的U值,计算公式为:U=(K1*S1+K2*S2)/S1+S2
    def get_avg_k(self):
        return round((self._glass.get_K() * self._glass.get_area() + self._window_frame.get_K() * self._window_frame.get_area()) / (self._glass.get_area() + self._window_frame.get_area()),4)

    # 计算窗框/窗总面积比值,计算公式为:R=S1/S2+S1
    def get_fw_ratio(self):
        return round(self._window_frame.get_area() / (self._glass.get_area() + self._window_frame.get_area()),4)
    
    # to_tensor
    def to_tensor(self): # shape (24,)

        area_tensor = tf.cast(tf.constant([self._area, self._orientation]), dtype=tf.float64)
        glass_tensor = tf.cast(tf.constant(self.get_glass().get_one_hot_code()), dtype=tf.float64)
        window_frame_tensor = tf.cast(tf.constant(self.get_window_frame().get_one_hot_code()), dtype=tf.float64)
        tensor = tf.concat([area_tensor, glass_tensor, window_frame_tensor], axis=0)
        # 数据类型转换为float64
        tensor = tf.cast(tensor, tf.float64)
       
        return tensor

    # tensor to window, shape (24,)
    def tensor_to_window(self, tensor):
        self._area = tensor[0]
        self._orientation = tensor[1]
        glass = Glass(warehouse=gmWarehouse,
                material_type=Glass(warehouse=gmWarehouse).get_material_type_by_one_hot_code(tensor[2:2+gmWarehouse.get_size()]))
        self._glass = glass

        window_frame = WindowFrame(warehouse=wfmWarehouse,
                material_type=WindowFrame(warehouse=wfmWarehouse).get_material_type_by_one_hot_code(tensor[2+gmWarehouse.get_size():2+gmWarehouse.get_size()+wfmWarehouse.get_size()]))
        self._window_frame = window_frame
        #window = Window(area=area, glass=glass, window_frame=window_frame, orientation=orientation)

        return self

    def get_empty_tensor():
        tensor = tf.constant( [0.0, 0] + 
                            [0] * gmWarehouse.get_size() + 
                            [0] * wfmWarehouse.get_size())
        # 数据类型转换为float64
        tensor = tf.cast(tensor, tf.float64)
       
        return tensor

    def get_window_features_len(self):
        return 2 + gmWarehouse.get_size() + wfmWarehouse.get_size()


class Wall:
    def __init__(self, area = 0.0, width=0.0, height=2.95,thickness=0.0, wi_material=None, orientation=0,window=None): 
        #thickness为墙体厚度,material为墙体材料,orientation为墙体朝向,如北墙,南墙等,
        self._area = round(area,4)
        self._width = round(width,4)
        self._height = round(height,4)
        self._thickness = round(thickness,4)
        self._material = wi_material
        self._orientation = orientation
        self._window = window
        if self._window is not None:
            self.add_window(window)
        self.wall_features_len = 2 + wimWarehouse.get_size() + 2 + gmWarehouse.get_size() + wfmWarehouse.get_size()

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
            raise self.WallDataError("The width of the window is greater than the length of the wall.")
            
        if window.get_height() > self._height:
            raise self.WallDataError("The height of the window is greater than the height of the wall.")

        if window.get_orientation() != self._orientation:
            raise self.WallDataError("The orientation of the window is not the same as the orientation of the wall.")
        return True

    def remove_window(self):
        if self._window is not None:
            self._window = None
        else:
            print("There is no window on this wall.")

    def calculate_area(self):
        self._area = round(self._width * self._height,4)
        return self._area

    def get_area(self):
        return self.calculate_area()

    def get_width(self):
        return self._width

    def set_width(self, width):
        self._width = round(width,4)

    def get_height(self):
        return self._height

    def set_height(self, height):
        self._height = round(height,4)

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

    def get_cost(self): #计算墙体成本,单位为元,计算公式为:墙体材料成本 + 窗户成本（如果有窗户，含玻璃和窗框
        if self._window is None:
            if self._material is not None:
                return self._material.get_cost()
            else:
                return 0.0

        return round(self._material.get_cost() + self._window.get_cost(),4)

    def get_avg_k(self):#计算墙体的平均K值,单位为W/(m2*K),计算公式为:U=(K1*S1+K2*S2)/S1+S2；其中K1为墙体材料的K值，S1为墙体面积，K2为窗户的K值，S2为窗户面积
        if self._window is None:
            if self._material is not None:
                return self._material.get_K()
            else:
                return 0.0

        return round((self._material.get_K() * (self.calculate_area()-self._window.get_area()) + self._window.get_avg_k() * self._window.get_area()) / self.calculate_area(),4)

    def get_ww_ratio(self):#计算墙/窗比值,计算公式为:R=S2/S1；其中S1为墙体面积，S2为窗户面积
        return round(self._window.get_area() / self.calculate_area(),4)  

    def to_json(self):
        if self._window is None:
            return {
            "orientation": self._orientation,   #墙体朝向
            "area": self._area,                 #墙体面积
            "wall_width": self._width,          #墙体宽度
            "wall_height": self._height,        #墙体高度
            "insulation_material": self._material.get_material_type(), #墙体材料
            "wall_avg_k": self.get_avg_k(),
            "window": None
            }
        else:
            wf_material = self._window.get_window_frame().get_material_type()
            wf_m_value = {'key':wf_material['key'],'type':wf_material['type'],'material':wf_material['material'],'price':wf_material['price'],'K':wf_material['K']}
            return {
                "orientation": self._orientation,   #墙体朝向
                "area": self._area,                 #墙体面积
                "wall_width": self._width,          #墙体宽度
                "wall_height": self._height,        #墙体高度
                "insulation_material": self._material.get_material_type(), #墙体材料
                "wall_avg_k": self.get_avg_k(),
                "window": {
                    "area": self._window.get_area(), #窗户面积
                    "window_wall_ratio": self.get_ww_ratio(), #窗/墙比值
                    "window_width": self._window.get_width(),                   #窗户宽度
                    "window_height": self._window.get_height(),                 #窗户高度
                    "orientation": self._window.get_orientation(),              #窗户朝向
                    "glass_material": self._window.get_glass().get_material_type(),  #窗户玻璃材料
                    "wfa_ratio": self._window.get_fw_ratio(),                  #窗框/窗比值
                    "glass_area": self._window.get_glass().get_area(),          #  玻璃面积
                    #"wf_material": self._window.get_window_frame().get_material_type(),#窗框材料
                    "wf_material": wf_m_value,#窗框材料
                    "wf_area": self._window.get_window_frame().get_area()       #窗框面积
                }
        }
    
    # json to wall, 从json数据中读取数据，创建墙体对象

    def json_to_wall(self, json_data):
        self._orientation = json_data["orientation"]
        self._area = json_data["area"]
        self._width = json_data["wall_width"]
        self._height = json_data["wall_height"]
        self._material = WallInsulation(area=round(json_data["area"],4),warehouse = wimWarehouse,material_type=json_data["insulation_material"])
        #self._material.set_material_type(json_data["insulation_material"])
        if json_data["window"] is not None:
            self._material.set_area(self._material.get_area()-json_data["window"]["area"])
            self._window = Window(area=json_data["window"]["area"])
            #self.reset_im_material_area()
            #self.add_window(Window(area=json_data["window"]["area"]))
            #self._window.set_area(json_data["window"]["area"])
            self._window.set_orientation(json_data["window"]["orientation"])
            self._window.set_width(json_data["window"]["window_width"])
            self._window.set_height(json_data["window"]["window_height"])
            self._window.set_glass(Glass(area=json_data["window"]["glass_area"],warehouse = gmWarehouse,
                                                            material_type=json_data["window"]["glass_material"]))
            #self._window.get_glass().set_material_type(json_data["window"]["glass_material"])
            self._window.set_window_frame(WindowFrame(area=json_data["window"]["wf_area"],warehouse = wfmWarehouse,
                                                            material_type=json_data["window"]["wf_material"]))
            #self._window.get_window_frame().set_material_type(json_data["window"]["wf_material"])
        else:
            self._window = None
        
        return self

    def to_json_cn(self):
        if self._window is None:
            return {
            "orientation(墙体朝向)": self._orientation,   #墙体朝向
            "area(墙体面积)": self._area,                 #墙体面积
            "wall_width(墙体宽度)": self._width,          #墙体宽度
            "wall_height(墙体高度)": self._height,        #墙体高度
            "insulation_material(墙体材料)": self._material.get_material_type() if self._material is not None else {}, #墙体材料
            "wall_avg_k(墙体平均导热系数)": self.get_avg_k(), #墙体平均导热系数
            "window": None
            }
        else:
            wf_material = self._window.get_window_frame().get_material_type()
            wf_m_value = {'key':wf_material['key'],'type':wf_material['type'],'material':wf_material['material'],'price':wf_material['price'],'K':wf_material['K']}
            return {
                "orientation(墙体朝向)": self._orientation,   #墙体朝向
                "area(墙体面积)": self._area,                 #墙体面积
                "wall_width(墙体宽度)": self._width,          #墙体宽度
                "wall_height(墙体高度)": self._height,        #墙体高度
                "insulation_material(墙体材料)": self._material.get_material_type(), #墙体材料
                "wall_avg_k(墙体平均导热系数)": self.get_avg_k(), #墙体平均导热系数
                "window": {
                    "area(窗户面积)": self._window.get_area(), #窗户面积
                    "window_wall_ratio(窗/墙比)": self.get_ww_ratio(), #窗/墙比值
                    "window_width(窗户宽度)": self._window.get_width(),                   #窗户宽度
                    "window_height(窗户高度)": self._window.get_height(),                 #窗户高度
                    "orientation(窗户朝向)": self._window.get_orientation(),              #窗户朝向
                    "glass_material(窗户玻璃材料)": self._window.get_glass().get_material_type(),  #窗户玻璃材料
                    "wfa_ratio(窗框/窗比)": self._window.get_fw_ratio(),                  #窗框/窗比值
                    "glass_area(玻璃面积)": self._window.get_glass().get_area(),          #  玻璃面积
                    #"wf_material(窗框材料)": self._window.get_window_frame().get_material_type(),#窗框材料
                    "wf_material(窗框材料)":wf_m_value,
                    "wf_area(窗框面积)": self._window.get_window_frame().get_area()       #窗框面积
                }
        }

    # to tensor

    def to_tensor(self): # shape=(34,)

        if self._window is None: # 是外墙，但是无窗户，只有墙体材料，需要补充一个空的窗户
            area_tensor = tf.cast(tf.constant([self._area, self._orientation]), dtype=tf.float64)
            material_tensor = tf.cast(tf.constant(self._material.get_one_hot_code()), dtype=tf.float64)
            window_tensor = tf.cast(Window.get_empty_tensor(), dtype=tf.float64)

            #print("Wall area tensor: ", area_tensor,"shape: ", area_tensor.shape)
            #print("Wall material tensor: ", material_tensor,"shape: ", material_tensor.shape)
            #print("Wall empty window tensor: ", window_tensor, "shape: ", window_tensor.shape)

            tensor = tf.concat([area_tensor, material_tensor, window_tensor], axis=0)

        else: # 是外墙，有窗户
            area_tensor = tf.cast(tf.constant([self._area, self._orientation]), dtype=tf.float64)
            material_tensor = tf.cast(tf.constant(self._material.get_one_hot_code()), dtype=tf.float64)
            window_tensor = self.get_window().to_tensor()
            window_tensor = tf.cast(window_tensor, dtype=tf.float64)
            
            #print("Wall area tensor: ", area_tensor,"shape: ", area_tensor.shape)

            #print("Wall material tensor: ", material_tensor,"shape: ", material_tensor.shape)
            #print("Wall window tensor: ", window_tensor, "shape: ", window_tensor.shape)

            
            tensor = tf.concat([area_tensor, material_tensor,window_tensor ],axis=0)

        #print("wall tensor: ", tensor)
        return tensor

    # tensor to wall, 从tensor中恢复墙体信息
    def tensor_to_wall(self,tensor): #, wimWarehouse, gmWarehouse, wfmWarehouse):
        #print("tensor: ", tensor)
        self._area = tensor[0]
        self._orientation = tensor[1]
        self._material = WallInsulation(material_type=WallInsulation(warehouse=wimWarehouse).get_material_type_by_one_hot_code(tensor[2:2+wimWarehouse.get_size()]))
        self._window = Window().tensor_to_window(tensor[2+wimWarehouse.get_size():])
        #return Wall(area, orientation, material, window)
        return self

    def add_empty_wall(): # [0,0] + 空材料，空窗户，空窗框

        area_tensor = tf.cast(tf.constant([0, 0]), dtype=tf.float64)
        material_tensor = tf.cast(WallInsulationMaterialWarehouse.get_empty_tensor(WallInsulationMaterialWarehouse()), dtype=tf.float64)
        window_tensor = tf.cast(Window.get_empty_tensor(), dtype=tf.float64)

        tensor = tf.concat([area_tensor, material_tensor,window_tensor ],axis=0)

        #print("empty wall tensor: ", tensor,"shape: ", tensor.shape)

        return tensor


    #@staticmethod
    def get_wall_features_len(self):
        #wall_features_len = 2 + wimWarehouse.get_size() + 2 + gmWarehouse.get_size() + wfmWarehouse.get_size()
        return self.wall_features_len


    # 定义一个异常类，用于处理墙数据错误
    class WallDataError(Exception):
        def __init__(self, ErrorInfo):
            super().__init__(self) # 初始化父类
            self.errorinfo = ErrorInfo
        def __str__(self):
            return self.errorinfo

class Room:

    ROOM_NUMBER = 12

    def __init__(self, name='room', type = 0, area = 0.00, length = 0.00, width = 0.00, height = 2.95,
        walls = ([None]) * 4,floor=None,doors=None,ceiling=None):
        #room_insulation为房间保温材料,door为房间门,window为房间窗户,ceiling为房间天花板,floor为房间地板
        # walls = [0,None] * 4, 0为内墙1为外墙，None为墙体，数组小标为墙体朝向，0为东墙，1为南墙，2为西墙，3为北墙
        self._name = name #房间名称
        self._type = type #房间类型
        self._area = round(area,4) #房间面积
        self._length = round(length,4) #房间长度
        self._width = round(width,4) #房间宽度
        self._height = round(height,4) #房间高度
        self._walls = walls #房间墙体
        self._floor = floor #房间地板
        self._doors = doors #房间门
        self._ceiling = ceiling #房间天花板

        if self._area > 0:      #如果房间面积大于0，则生成房间长宽
            if self._length == 0 and self._width == 0:
                self.generate_room_wh(self._area)
 

    def get_name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    def get_type(self):
        return self._type
    
    def set_type(self, type):
        self._type = type

    def get_area(self): # 2023.7.3修改，增加了对房间面积的计算
        self._area = round(self._length * self._width,4)
        return self._area
    
    def set_area(self, area):
        self._area = round(area,4)

    def generate_room_wh(self, room_area):
        ratio = np.random.uniform(0.618, 1.618) # 生成长宽比，范围为0.618-1.618，符合黄金分割
        length = (room_area * ratio)**0.5
        width = room_area / length
        self.set_length(length)
        self.set_width(width)
    
    def get_length(self):
        return self._length
    
    def set_length(self, length):
        self._length = round(length,4)
    
    def get_width(self):
        return self._width

    def set_width(self, width):
        self._width = round(width,4)

    def get_height(self):
        return self._height

    def set_height(self, height):
        self._height = height
        
    def get_walls(self):
        return self._walls

    def set_walls(self, walls):
        self._walls = walls

    def add_wall(self, orientation,wall):
        if self.check_wall_(orientation,wall):
            self._walls[orientation] = wall

    def remove_wall(self, orientation,wall):
        #if self.check_wall_(orientation,wall):
        self._walls[orientation] = None
        

    # 判断墙数据是否合乎房间数据
    def check_wall_(self, orientation,wall):

        #print("wall width = ",wall.get_width(),"room width = ",self.get_length())

        if orientation == Orientation.EAST or orientation == Orientation.WEST:# 0为东墙，2为西墙

            if wall.get_width()- 1e-3 > self.get_width():# 判断墙的长度是否合法
            #if wall.get_width() > self.get_width():# 判断墙的长度是否合法
                    raise self.RoomDataError("wall width is invalid, wall width is %f, room width is %f" % (wall.get_width(), self.get_width()))

        if orientation == Orientation.SOUTH or orientation == Orientation.NORTH:# 1为南墙，3为北墙
            
            if wall.get_width()- 1e-3 > self.get_length():# 判断墙的长度是否合法
            #if wall.get_width() > self.get_length():# 判断墙的长度是否合法
                raise self.RoomDataError("wall width is invalid, wall width is %f, room width is %f" % (wall.get_width(), self.get_length()))
        
        if wall.get_height()- 1e-3 > self.get_height():# 判断墙高度是否合法
        #if wall.get_height() > self.get_height():# 判断墙高度是否合法

            raise self.RoomDataError("wall height is invalid, wall height is %f, room height is %f" % (wall.get_height(), self.get_height()))

        return True

    # 判断房间是否符合规范
    def valid_wall_windows_ratio(self, min_ratio,max_ratio):
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
        total_wall_area = 0
        for wall in self._walls:
            if wall is not None and wall._material is not None:
                total_wall_area += wall.calculate_area()
        return round(total_wall_area,4)

    def get_total_window_area(self):
        total_window_area = 0
        for wall in self._walls:
            if wall is not None and wall.get_window() is not None:
                total_window_area += wall.get_window().get_area()
        return round(total_window_area,4)

    def get_total_wall_num(self):
        total_wall_num = 0
        for wall in self._walls:
            if wall is not None and wall._material is not None:
                total_wall_num += 1
        return total_wall_num

    def get_total_window_num(self):
        total_window_num = 0
        for wall in self._walls:
            if wall is not None and wall.get_window() is not None:
                total_window_num += 1
        return total_window_num
    
    # 定义一个异常类，用于处理房间数据错误
    class RoomDataError(Exception):
        def __init__(self, ErrorInfo):
            super().__init__(self) # 初始化父类
            self.errorinfo = ErrorInfo
        def __str__(self):
            return self.errorinfo

    # 外墙玻璃，窗框，窗户 总成本
    def get_cost(self):
        self._cost = 0
        for wall in self._walls:
            #if wall is not None and wall[0] == 1 and wall[1] is not None:
            if wall is not None and wall._material is not None:
                self._cost += wall.get_cost()
        return round(self._cost,2)

    def get_avg_k(self):

        self._avg_k = 0
        #print ("self._walls = ",self._walls)
        t_K = 0
        t_a = 0
        for wall in self._walls:
        
            #if wall is not None and wall[0] == 1 and wall[1] is not None:
            if wall is not None:
                t_K += wall.get_avg_k() * wall.get_area()
                t_a += wall.get_area()

        if t_a == 0:
            self._avg_k = 0
        else:
            self._avg_k = t_K / t_a

        return round(self._avg_k,4)


    def get_ww_ratio(self):# 计算墙/窗比值,计算公式为:R=S2/S1；其中S1为墙体面积，S2为窗户面积

        self._ww_ratio = 0
        t_w = 0
        for wall in self._walls:
            if wall is not None  and wall.get_window() is not None:
                t_w += wall.get_window().get_area() 
        
        if self.get_total_wall_area() == 0:
            self._ww_ratio = 0
        else:
            self._ww_ratio = t_w / self.get_total_wall_area()

        return round(self._ww_ratio,4)

    def get_total_cost(self): # 计算房间总成本
        self._cost = 0
        for wall in self._walls:
            if wall is not None and wall._material is not None:
                self._cost += wall.get_cost()
        return round(self._cost,2)

    # 计算外墙材料总成本
    def get_totoal_material_cost(self):
        im_cost = 0
        for wall in self._walls:
            if wall is not None and wall._material is not None:
                im_cost += wall.get_material_cost()
        return round(im_cost,4)

    # 计算外墙玻璃总成本
    def get_total_glass_cost(self):
        glass_cost = 0
        for wall in self._walls:
            if wall is not None and wall._material is not None:
                glass_cost += wall.get_glass_cost()
        return round(glass_cost,4)

    # 计算外墙窗框总成本
    def get_total_frame_cost(self):
        frame_cost = 0
        for wall in self._walls:
            if wall is not None and wall._material is not None:
                frame_cost += wall.get_frame_cost()
        return round(frame_cost,4)

    # 计算外墙和房屋面积比值
    def get_wr_ratio(self):
        return round(self.get_total_wall_area() / self.get_area(),4)
    
    def to_json(self):
        return {
            "room_name":self._name,     # 房间名称
            "room_type": self._type,    # 房间类型
            "room_area": self._area,    # 房间面积
            "room_length": self._length,# 房间长度
            "room_width": self._width,  # 房间宽度
            "room_height": self._height, # 房间高度
            "total_wall_area": self.get_total_wall_area(), # 房间外墙面积
            "total_window_area": self.get_total_window_area(), # 房间窗户面积
            "ww_ratio": self.get_ww_ratio(), # 房间外墙/窗户比值
            'walls': [wall.to_json() for wall in self._walls if wall is not None and wall._area > 0.0 ],
            "total_cost_im": self.get_totoal_material_cost(), # 房间外墙材料总成本
            "total_cost_g": self.get_total_glass_cost(), # 房间外墙玻璃总成本
            "total_cost_wf": self.get_total_frame_cost(), # 房间外墙窗框总成本
            "total_cost": self.get_total_cost(), # 房间总成本
            "total_avg_k": self.get_avg_k() # 房间平均导热系数
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
            self.add_wall(_wall.get_orientation(),_wall)
            #self._walls.append(Wall().json_to_wall(wall))

        # 补充其它空墙
        for i in [Orientation.EAST, Orientation.WEST, Orientation.SOUTH, Orientation.NORTH]:
            if self._walls[i] is None:
                self.add_wall(i,Wall())

        return self
    '''    
    def json_to_room(self, json_data):
        self._type = json_data['room_type']
        self._area = json_data['room_area']
        self._length = json_data['room_length']
        self._width = json_data['room_width']
        self._height = json_data['room_height']
        self._walls = []
        for wall in json_data['walls']:
            self._walls.append(Wall().json_to_wall(wall))
        return self
        '''
    
    def to_json_cn(self):
        return {
            "room_name(房间名称)":self._name,     # 房间名称
            "room_type(房间类型)": self._type,    # 房间类型
            "room_area(房间面积)": self._area,    # 房间面积
            "room_length(房间长度)": self._length,# 房间长度
            "room_width(房间宽度)": self._width,  # 房间宽度
            "room_height(房间高度)": self._height, # 房间高度
            "total_wall_area(房间外墙面积)": self.get_total_wall_area(), # 房间外墙面积
            "total_window_area(房间窗户面积)": self.get_total_window_area(), # 房间窗户面积
            "ww_ratio(房间外墙/窗户比值)": self.get_ww_ratio(), # 房间外墙/窗户比值
            #'walls': [wall.to_json_cn() for wall in self._walls if wall is not None],
            'walls': [wall.to_json_cn() for wall in self._walls if wall is not None and wall._area > 0.0 ],
            "total_cost_im(房间外墙材料总成本)": self.get_totoal_material_cost(), # 房间外墙材料总成本
            "total_cost_g(房间外墙玻璃总成本)": self.get_total_glass_cost(), # 房间外墙玻璃总成本
            "total_cost_wf(房间外墙窗框总成本)": self.get_total_frame_cost(), # 房间外墙窗框总成本
            "total_cost(房间总成本)": self.get_total_cost(), # 房间总成本
            "total_avg_k(房间平均导热系数)": self.get_avg_k() # 房间平均导热系数
        }



    # 转换为张量
    # 获取空房间张量 shape=(140,), 140=4+4*34,
    # target = [cost,k] 代表房间cost和 k值
    def to_tensor(self):# 转换为张量 shape=(140,)

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

        target = tf.cast(tf.constant([self.get_total_cost(), self.get_avg_k()]), dtype=tf.float64)
        #print("room tensor ", tensor,"shape ", tensor.shape, "target ", target)
        #print("room tensor shape ", tensor.shape, "target shape ", target.shape)
        return tensor , target

    # tensor to room
    # 用于从张量中恢复房间对象
    def tensor_to_room(self, tensor):
        self._type = tensor[0]
        self._area = tensor[1]
        #self._length = math.sqrt(self._area)
        #self._width = self._length
        #self._height = 3.0
        self._walls = [None]*4
        f_len = Wall().get_wall_features_len()

        for i in range(4):
            self.add_wall(i,
                            Wall().tensor_to_wall(tensor[2+i*f_len:2+(i+1)*f_len])
                        )

        return self

    # 获取空房间张量 shape=(140,), 140=4+4*34,
    # target = [0,0] 代表房间cost和 k值
    @staticmethod
    def get_empty_room_tensor(): 
        tensor = tf.cast(tf.constant([0,0,0,0]), dtype=tf.float64)
        

        for i in range(4):
            tensor = tf.concat([tensor, Wall.add_empty_wall()], axis=0)
        
        target = tf.cast(tf.constant([0,0]), dtype=tf.float64)

        return tensor , target

    #@staticmethod
    def get_room_features_len(self):
        return self.get_empty_room_tensor()[0].shape[0]


# 房屋类
class House:

    wall_features_len = Wall().get_wall_features_len()
    room_features_len = Room().get_room_features_len()
    ml_logger.info(f"wall_features_len {wall_features_len} room_features_len {room_features_len}")
    #HOUSE_FEATURES = 1680
    #TARGET = 26
    HOUSE_FEATURES = _ss.get_max_num_rooms() * room_features_len
    TARGET = _ss.get_max_num_rooms() * 2 + 2

    ml_logger.info(f"HOUSE_FEATURES {HOUSE_FEATURES} TARGET {TARGET}")
    

    def __init__(self, name='House', area=0.0, height=2.95,rooms=[]):
        self._name = name
        self._area = area
        self._height = height
        self._rooms = rooms
        self._cost_view ={}

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

    def get_rooms(self):
        return self._rooms

    def set_rooms(self, rooms):
        self._rooms = rooms

    def add_room(self, room):
        self._rooms.append(room)

    def update_room(self, room, target_room):
        index = self._rooms.index(target_room)
        self._rooms[index] = room
    
    def del_room(self, room):
        self._rooms.remove(room)

    def get_room_num(self):
        return len(self._rooms)
    
    def get_total_cost(self):
        self._cost = 0
        for room in self._rooms:
            self._cost += room.get_cost()
        return round(self._cost,2)

    def get_total_area(self):# 计算房屋面积
        self._total_area = 0
        for room in self._rooms:
            self._total_area += room.get_area()
        return round(self._total_area,4)

    def get_total_window_area(self):# 计算房屋窗户面积
        self._total_window_area = 0
        for room in self._rooms:
            self._total_window_area += room.get_total_window_area()
        return round(self._total_window_area,4)

    def get_total_wall_num(self): # 计算房屋外墙数量
        self._total_wall_num = 0
        for room in self._rooms:
            self._total_wall_num += room.get_total_wall_num()
        return self._total_wall_num

    def get_total_window_num(self): # 计算房屋窗户数量
        self._total_window_num = 0
        for room in self._rooms:
            self._total_window_num += room.get_total_window_num()
        return self._total_window_num


    def get_total_wall_area(self): # 计算房屋外墙面积
        self._total_wall_area = 0
        for room in self._rooms:
            self._total_wall_area += room.get_total_wall_area()
        return round(self._total_wall_area,4)

    def get_avg_ww_ratio(self): # 计算平均墙/窗比值
        self._avg_ww_ratio = 0
        for room in self._rooms:
            self._avg_ww_ratio += room.get_ww_ratio() * room.get_area()
        self._avg_ww_ratio = self._avg_ww_ratio / self.get_total_area()
        return round(self._avg_ww_ratio,4)


    def get_avg_k(self):# 计算平均传热系数
        self._avg_k = 0
        for room in self._rooms:
            self._avg_k += room.get_avg_k() * room.get_area()
        self._avg_k = self._avg_k / self.get_total_area()
        return round(self._avg_k,4)

    def get_total_area(self): # 计算房屋总面积
        self._area = 0
        for room in self._rooms:
            self._area += room.get_area()
        return round(self._area,4)


    # get cost view,汇总不同型号材料所用的面积、数量、价格、总价
    # 包括 glass,window_frame,wall,wall_insulation
    # return {'material':{'area':,'price':,'cost':}}
    def get_cost_view(self):
        #pdb.set_trace()
        self._cost_view = {}
        for room in self._rooms:
            for wall in room.get_walls():
                # 外墙
                if wall is None:
                    continue
                material = wall.get_material()
                self.add_cost_view(material)
                window = wall.get_window()
                if window is not None:
                    # 玻璃
                    material = window.get_glass()
                    self.add_cost_view(material)
                    # 窗框
                    material = window.get_window_frame()
                    self.add_cost_view(material)
        return self._cost_view

    # add material to cost_view
    def add_cost_view(self,material):
        #pdb.set_trace()
        if material is None:
            return
        if isinstance(material,Glass):
            material_type = material.get_material_type()['descriptions']
        elif isinstance(material,WallInsulation):
            material_type = material.get_material_type()['name']
        elif isinstance(material,WindowFrame):
            material_type = material.get_material_type()['type']
        else:
            material_type = material.get_material_type()

        if any(material_type == key for key in self._cost_view):

            self._cost_view[material_type]['area'] = round(self._cost_view[material_type]['area'] + material.get_area(),4)
            self._cost_view[material_type]['cost'] = round(self._cost_view[material_type]['cost'] + material.get_cost(),2)
            #self._cost_view[material_type]['area'] += round(material.get_area(),4)
            #self._cost_view[material_type]['cost'] += round(material.get_cost(),2)
        else:
            key = str(material_type)  # Convert the dictionary to a string representation
            self._cost_view[key] = {
                'area': round(material.get_area(),4),
                'price': material.get_price(),
                'cost': round(material.get_cost(),2)
            }

    def to_json(self):
        return {
            'h_name': self._name, # 房屋名称
            'h_total_area': self.get_total_area(), # 房屋总面积
            'h_total_room_num': self.get_room_num(), # 房屋房间数量
            'h_total_wall_num': self.get_total_wall_num(), # 房屋外墙数量
            'h_total_wall_area': self.get_total_wall_area(), # 房屋外墙面积
            'h_total_window_area': self.get_total_window_area(), # 房屋窗户面积
            'h_total_window_num': self.get_total_window_num(), # 房屋窗户数量
            'h_height': self._height, # 房屋高度
            'h_total_avg_k': self.get_avg_k(), # 房屋平均导热系数
            'h_total_avg_ww_ratio': self.get_avg_ww_ratio(), # 房屋平均墙/窗比值
            'h_total_cost': self.get_total_cost(), # 房屋总材料成本
            'rooms': [room.to_json() for room in self._rooms]
        }

    def json_to_house(self,json_data):
        #house = House()
        self.set_name(json_data['h_name'])
        self.set_height(json_data['h_height'])
        rooms = []
        for room in json_data['rooms']:
            rooms.append(Room().json_to_room(room))
        self.set_rooms(rooms)
        #self.set_rooms([Room.json_to_room(room) for room in json['rooms']])
        return self
    

    def to_json_cn(self):
        return {
            'h_name [房屋名称]': self._name, # 房屋名称
            'h_total_area [房屋总面积]': self.get_total_area(), # 房屋总面积
            'h_total_room_num [房屋房间数量]': self.get_room_num(), # 房屋房间数量
            'h_total_wall_num [房屋外墙数量]': self.get_total_wall_num(), # 房屋外墙数量
            'h_total_wall_area [房屋外墙面积]': self.get_total_wall_area(), # 房屋外墙面积
            'h_total_window_area [房屋窗户面积]': self.get_total_window_area(), # 房屋窗户面积
            'h_total_window_num [房屋窗户数量]': self.get_total_window_num(), # 房屋窗户数量
            'h_height [房屋高度]': self._height, # 房屋高度
            'h_total_avg_k [房屋平均导热系数]': self.get_avg_k(), # 房屋平均导热系数
            'h_total_avg_ww_ratio [房屋平均墙/窗比值]': self.get_avg_ww_ratio(), # 房屋平均墙/窗比值
            'h_total_cost [房屋总材料成本]': self.get_total_cost(), # 房屋总材料成本
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
            json.dump(self.to_json(), json_file, ensure_ascii=False, indent=4, cls=self.CustomJSONEncoder)

    def save_to_json_cn(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(self.to_json_cn(), json_file, ensure_ascii=False, indent=4, cls=self.CustomJSONEncoder)

    # room_tensors's shape = (1680,) = ROOM_NUMBER* room_tensor's shape(140,) + 4,
    # target's shape ROOM_NUMBER*2 + 2 = 26
    
    def to_tensor(self): 
        #pdb.set_trace()
        room_tensors = []
        targets = []

        for room in self._rooms:
            if room is not None:
                r_tensor, r_target = room.to_tensor()

                room_tensors.append(r_tensor)
                targets.append(r_target)

        for i in range(Room.ROOM_NUMBER - len(room_tensors)):
            empty_tensors , empty_targets = Room.get_empty_room_tensor()

            room_tensors.append(empty_tensors)
            targets.append(empty_targets)

        targets.append([self.get_total_cost(),self.get_avg_k()])

        room_tensors = tf.concat(room_tensors, axis=0)
        targets = tf.concat(targets, axis=0)
        targets = tf.cast(targets, tf.float64)

        return room_tensors, targets

    # tensor to house
    def tensor_to_house(self,room_tensors,targets):
        rooms = []
        for i in range(Room.ROOM_NUMBER):
            room = Room()
            room.tensor_to_room(room_tensors[i*room.get_room_features_len():(i+1)*room.get_room_features_len()])
                                #targets[i*2:(i+1)*2])
                                #targets[i*Room.ROOM_TARGETS_LEN:(i+1)*Room.ROOM_TARGETS_LEN])
            rooms.append(room)
        self.set_rooms(rooms)
        self.get_total_cost()
        self.get_avg_k()
        #self.set_total_cost(targets[-2])
        #self.set_avg_k(targets[-1])
        return self

    wall_features_len = Wall().get_wall_features_len()
    room_features_len = Room().get_room_features_len()
    # 根据tensor 中的index 找到对应的feature name
    def find_feature_name(self,index):

        point = index % self.room_features_len #Room.ROOM_NUMBER
        #ml_logger.info(f"--------------index {index} ,point {point} ")
        if point < 4: # 
            #print("point <4 ", point )
            return "No mutation allowed"

        #if point >= 4  and  point < 4 + wimWarehouse.get_size():
        #    return "wall_material"
        else:
            bpoint = point
            point = point - 4

            point = point % self.wall_features_len #Wall().get_wall_features_len()

            #ml_logger.info(f"point wall-disct index-bpoint-point,{self.wall_features_len}-{index}-{bpoint}-{point}")

            #ml_logger.info(f"############# index {index} ,point {point} ")

            if point >= 2 and point < 2 + wimWarehouse.get_size():
                return "wall_material"
            elif point == 2 + wimWarehouse.get_size() : #and point < 2 + wimWarehouse.get_size() + 2:
                return "window_size"
            elif point >= 2 + wimWarehouse.get_size() + 2 and point < 2 + wimWarehouse.get_size() + 2 + gmWarehouse.get_size():
                return "glass_material"
            elif point >= 2 + wimWarehouse.get_size() + 2 + gmWarehouse.get_size():
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
        
        # 创建一个房屋对象,房屋的高度为标准层高
        house = House(name=name, height= _ss.get_standard_floor_height(), rooms=_ss.get_random_rooms()) 

        for index  in house._rooms:

            (min_area, max_area, min_wall_num, max_wall_num,
            required, min_room_num, max_room_num, min_window_num, max_window_num) = _ss.get_room_encoding_info(index)

            room_area = np.random.uniform(min_area, max_area) # 生成一个房间面积
            min_window_wall_ratio, max_window_wall_ratio = _ss.get_room_wwr_range(index)# 5.19 改为独立获取每个房间的窗墙比
            
            # 创建一个房间对象,walls=[0,None] * 4表示房间的四面墙都没有窗户
            room = Room(name=f"Room {index} ", type=index, 
                        area= room_area, height = house.get_height(), 
                        walls=[None] * 4, floor=None, doors=[], ceiling=None) 

            # 计算房间的length 和 width
            room.generate_room_wh(room_area)
            #print(f"room {index} area: {room_area}, length: {room.get_length()}, width: {room.get_width()}")
            
            # 生成一个房间的墙的数量,min_wall_num和max_wall_num是墙的数量的范围
            wall_num = np.random.randint(min_wall_num, max_wall_num +1) 
            
            if wall_num == 0: # 如果墙的数量为0,则不需要创建新的外墙
                house.update_room(room,index) 

            #    house.add_room(room) # 将房间添加到房屋中
                continue
            #生成一个有序的断点列表(breakpoints)，然后计算相邻断点之间的差异，并将这些差异存储在一个列表(differences)中            
            breakpoints = sorted([0] + list(np.random.rand(wall_num - 1)) + [1])
            differences = [breakpoints[i+1] - breakpoints[i]
                    for i in range(len(breakpoints) - 1)]
            
            or_keys = self.orientation.get_orientation_keys()
            
            valid_windows = False
            invalid = 0
            while not valid_windows:
                for i, diff in enumerate(differences):

                    select_or = random.choice(or_keys) # 生成一个随机的方向
                    # 生成一个墙的宽度，如果方向为东西方向，则墙的宽度为房间的宽度，否则为房间的长度
                    wall_width = room.get_width() if select_or in [Orientation.EAST, Orientation.WEST] else room.get_length() 

                    # 生成的墙中 80%的概率为完全为外墙，也存在部分外墙可能性
                    exterior_ratio = np.random.rand()
                    if exterior_ratio < 0.8:
                        exterior_ratio = 1  
                    # 生成一个墙的面积
                    r_height = room.get_height()
                    #print("r_height: ",r_height, "wall_width :" ,wall_width,"exterior_ratio :",exterior_ratio)
                    exterior_wall_area = round(wall_width * exterior_ratio * r_height,4 )
                    # 生成一个墙对象
                    wall = Wall(    area = exterior_wall_area,  # 生成一个墙的面积
                                    width = wall_width, height=r_height,thickness=0.0, # 生成一个墙的宽度,高度,厚度 
                                    wi_material = WallInsulation( area = exterior_wall_area, warehouse = wimWarehouse, material_type = wimWarehouse.get_random_material() ),  # 生成一个墙的材料,缺省 外墙材料面积为墙的面积，在生成窗后，减除窗的面积
                                    orientation = select_or, # 生成一个墙的朝向
                                    window = None # 生成一个墙的窗户
                                ) 
                    # 以 50% 的概率为墙面分配一个窗户
                    window_wall_ratio = 0.0
                    if np.random.rand() < 0.5:

                        # 根据朝向，查找window_wall_ratio_limit
                        max_window_wall_ratio = _ss.get_window_wall_ratio_limit(wall.get_orientation())
                        # 如果窗户的面积比例大于最大的窗户面积比例，则将窗户面积比例设置为最大的窗户面积比例
                        if max_window_wall_ratio <= min_window_wall_ratio:
                            min_window_wall_ratio = min_window_wall_ratio - 0.01

                        # 生成一个窗户的面积比例
                        window_wall_ratio = np.random.uniform(
                            min_window_wall_ratio, max_window_wall_ratio)
                        # 生成一个窗户的面积
                        window_area = round(wall.get_area() * window_wall_ratio, 4)

                        # 根据窗户的面积和宽高比约束来计算窗户的宽度和高度
                        # 如果窗户的宽度或高度超过了墙的宽度或高度，则按比例缩小窗户的尺寸,
                        # 保证窗户不超过墙的尺寸,保证窗户的宽高比在0.618~1.618之间
                        aspect_ratio = np.random.uniform(0.618, 1.618)
                        window_width = np.sqrt(window_area * aspect_ratio)
                        window_height = window_area / window_width

                        if window_width > wall.get_width():
                            window_width = wall.get_width()
                            window_height = window_area / window_width
                        if window_height > wall.get_height():
                            window_height = wall.get_height()
                            window_width = window_area / window_height
                    
                        # 生成一个窗户对象
                        window = Window(area=window_area, width=window_width,height=window_height,orientation=wall.get_orientation()) 
                        # 生成一个窗框对象
                        # 生成一个窗框的材料,并将窗框的材料设置到窗框上,并计算窗框的面积比例
                        wfa_ratio, wf_material = wfmWarehouse.get_best_wf(window_width, window_height)
                        # 将窗框的材料设置到窗框上
                        window_frame = WindowFrame(area=window_area*wfa_ratio,warehouse = wfmWarehouse,material_type=wf_material)
                        # 生成一个窗框对象，将窗框添加窗户到上
                        window.set_window_frame(window_frame) 
                        # 生成一个玻璃对象，将玻璃添加到窗户上
                        glass = Glass(area=window_area*(1-wfa_ratio),warehouse = gmWarehouse,material_type=gmWarehouse.get_random_material())
                        window.set_glass(glass) 
                        # 将窗户添加到墙上
                        wall.add_window(window) 
                        # 把材料，窗户，框的面积set 完整
                        #

                    # 将墙添加到房间中
                    room.add_wall(select_or,wall) 

                # 如果房间的墙面和窗户面积比例合法，则退出循环
                if room.valid_wall_windows_ratio( min_window_wall_ratio, max_window_wall_ratio):
                    valid_windows = True
                else:
                    invalid = invalid + 1
                if invalid > 500:
                    raise ValueError("Invalid wwr too times")

            # 将房间添加到房屋中
            house.update_room(room,index) 

        return house

def main():
# 生成一个房屋生成器对象
    hc = HousesCreator(1)
        # 生成一个房屋对象
    houses = hc.make_houses()

    for house in houses:
        #house.save_to_json(f"./houses_json/house_{house._name}.json")
        #house.save_to_json_cn(f"./houses_json/cn_house_{house._name}.json")

        #print("-------------------------------------------------------------------")
        #print("house tensor",house.to_tensor())
        #houses_features ,target = house.to_tensor()
        #print("houses_features ",houses_features,"shape *****************:",houses_features.shape)
        #print("target ",target,"shape *****************:",target.shape)
        #print("*******************************************************************")
        print(house.get_cost_view())



        #print(target)
        # 打印房屋信息
    #print(houses)
def load_json_to_house(json_file_path):
    json_data = None
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
    house = House()
    house.json_to_house(json_data)

    house.save_to_json(f"{json_file_path}_update.json")

if __name__ == "__main__":
    main()
    #load_json_to_house("./houses_json/training_dataset_json/20230611-135209-325994")
