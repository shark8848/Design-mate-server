import tensorflow as tf
import sys
sys.path.append("..")
from apocolib import RpcProxyPool
from ml_server_v2.PolyRegressor import PolyRegressor
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from apocolib.GPUConfigurator import GPUConfigurator

#configurator = GPUConfigurator(use_gpu=False, gpu_memory_limit=None)
#configurator = GPUConfigurator(use_gpu=True, gpu_memory_limit=2048)
#configurator.configure_gpu()
#tf.device(configurator.select_device())

class MaterialWarehouse:
    def __init__(self):
        self.materials = {}
        self.pool = RpcProxyPool.RpcProxyPool()
        self.materials_loaded = False
    
    def load_materials(self, materials):
        self.materials = materials
        self.materials_loaded = True
        #pass
    def get_materials(self):
        return self.materials
    
    def get_material(self, material_id):
        return self.materials[material_id]

    def get_material_info(self, material_id, *args):
        return self.materials[material_id][args[0]]
    
    def get_materials_info(self, *args):
        return [self.materials[material_id][args[0]] for material_id in self.materials]

    def get_materials_info_by_ids(self, material_ids, *args):
        return [self.materials[material_id][args[0]] for material_id in material_ids]

    # 随机选取一个材料
    #@staticmethod
    def get_random_material(self):
        import random
        return random.choice(self.materials)

    # 查询指定材料的ID的某个参数
    #@staticmethod
    def get_material_parameter(self, material_id, parameter):
        return self.materials[str(material_id)][parameter]

    def get_size(self):
        return len(self.materials)

    def find_key(self,value):
        for key, val in self.materials.items():
            if val == value:
                return key
        return None  # 如果未找到匹配的键，则返回None或其他适当的值

    def get_empty_tensor(self):# 生成一个空的张量
        # 生成一个1维度的数量为 self.get_size() 的全0张量
        return tf.zeros(shape=(self.get_size(),), dtype=tf.float64)


class GlassMaterialWarehouse(MaterialWarehouse):
    def __init__(self):
        super().__init__()
        self.load_materials(self.load_glass_materials())

    def load_glass_materials(self):
        rpc_proxy = self.pool.get_connection()
        error_code,data,error_msg = rpc_proxy.get_glass_materialsService.get_glass_materials()
        self.pool.put_connection(rpc_proxy)
        #print("gm_materials ",data)

        converted_dict = {}

        for item in data:
            item_id = item['id']
            converted_dict[item_id] = {
                'key': item_id,
                'descriptions': item['description'],
                'S_D': item['S_D'],
                'thickness': int(item['thickness']),
                'coating': item['silver_plated'],
                'hollow_material': item['hollow_material'],
                #'price': item['price'],
                'price': round(float(item['price']),2),
                'K': item['K']
            }

        #print("converted_dict ",converted_dict)
        return converted_dict
        '''
        return {
 
            0: {'descriptions': '单层LOW-E玻璃',               'S_D': 1, 'thickness': 5, 'coating': 1, 'hollow_material': 0,  'price': 41.04,  'K': 3.57},
            1: {'descriptions': '单层LOW-E玻璃',               'S_D': 1, 'thickness': 6, 'coating': 1, 'hollow_material': 0,  'price': 46,     'K': 3.57},
            2: {'descriptions': '单层LOW-E玻璃',               'S_D': 1, 'thickness': 8, 'coating': 1, 'hollow_material': 0,  'price': 57.17,  'K': 3.57},
            3: {'descriptions': '单层LOW-E玻璃',               'S_D': 1, 'thickness': 12,'coating': 1, 'hollow_material': 0,  'price': 64.58,  'K': 3.20},    
            4: {'descriptions': '5中透光Low-E+12A+5透明',      'S_D': 2, 'thickness': 5, 'coating': 1, 'hollow_material': 12, 'price': 113.06, 'K': 1.8696},
            5: {'descriptions': '6中透光Low-E+12A+6透明',      'S_D': 2, 'thickness': 6, 'coating': 1, 'hollow_material': 12, 'price': 123.82, 'K': 1.8628},
            6: {'descriptions': '8中透光Low-E+12A+8透明',      'S_D': 2, 'thickness': 8, 'coating': 1, 'hollow_material': 12, 'price': 145.32, 'K': 1.8494},
            7: {'descriptions': '5中透光Low-E+12A+5透明钢化',  'S_D': 2, 'thickness': 5, 'coating': 1, 'hollow_material': 12, 'price': 125.44, 'K': 1.8696},
            8: {'descriptions': '6中透光Low-E+12A+6透明钢化', 'S_D': 2, 'thickness': 6, 'coating': 1, 'hollow_material': 12, 'price': 137.98, 'K': 1.8628},
            9: {'descriptions': '8中透光Low-E+12A+8透明钢化', 'S_D': 2, 'thickness': 8, 'coating': 1, 'hollow_material': 12, 'price': 166.56, 'K': 1.8494}

        }
        '''


class WallInsulationMaterialWarehouse(MaterialWarehouse):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        super().__init__()
        if not self.materials_loaded:
            self.load_materials(self.load_wall_insulation_materials())

    def load_wall_insulation_materials(self):

        rpc_proxy = self.pool.get_connection()
        error_code,data,error_msg = rpc_proxy.get_wall_materialsService.get_wall_materials()
        self.pool.put_connection(rpc_proxy)
        #print("wi_materials ",data)

        converted_dict = {}

        for i, item in enumerate(data):
            converted_dict[i] = {
                'key': i,
                'name': item['description'],
                'thickness': int(item['thickness']),
                #'price': item['price'],
                'price': round(float(item['price']),2),
                'K': item['K'],
                'level': item['level']
            }

        self.materials = converted_dict
        self.materials_loaded = True

        return self.materials


class WindowFrameMaterialWarehouse(MaterialWarehouse):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        super().__init__()
        self.pr = PolyRegressor()

        if not self.materials_loaded:
            self.load_materials(self.load_window_frame_materials())

    def load_window_frame_materials(self):

        rpc_proxy = self.pool.get_connection()
        error_code,data,error_msg = rpc_proxy.get_wf_materialsService.get_wf_materials()
        self.pool.put_connection(rpc_proxy)
        #print("wi_materials ",data)

        converted_dict = {}
        for i, item in enumerate(data):
            converted_dict[i] = {
                'key': i,
                'type': f"WF-{i+1:03}",
                'material': item['wf_material'],
                'area': item['window_area'] + item['wf_area'],
                'width': item['profile_section'],
                'height': item['window_area'] / item['profile_section'],
                'frame_area': item['wf_area'],
                'price': round(float(item['price']),2),
                'window_frame_area_ratio': item['window_frame_area_ratio'],
                'K': item['K']
            }

        self.materials = converted_dict
        self.materials_loaded = True
        #print(self.materials)

        return self.materials
    #@staticmethod
    def get_best_wf(self, target_width, target_height):
        best_fit = None
        best_distance = float('inf')
        best_key = None
        for window_key in self.materials:
            window = self.materials[window_key]
            width = window['width']
            height = window['height']
            distance = ((width - target_width) ** 2 + (height - target_height) ** 2) ** 0.5

            # 只考虑距离更接近的窗户
            if distance < best_distance:
                best_distance = distance
                best_fit = window
                best_key = window_key

        # 返回 window_frame_area_ratio, window_frame_price, window_frame_k
        if best_fit is None:
            return 0.0,{}
            #0.0,0.0,{}
            #return 0.0, 0.0, 0.0
        else:
            return round(self.pr.predict(target_width*target_height),4), {**best_fit }
'''
class WallInsulationMaterialWarehouse(MaterialWarehouse):
    def __init__(self):
        super().__init__()
        self.materials_loaded = False
        self.load_materials(self.load_wall_insulation_materials())
    def load_wall_insulation_materials(self):
        return {
            0: {'name': '20mm厚度不燃型复合膨胀聚苯乙烯保温板 ( A 级)', 'thickness': 20, 'price': 63.85, 'K': 0.065, 'level': 1},
            1: {'name': '25mm厚度不燃型复合膨胀聚苯乙烯保温板 ( A 级)', 'thickness': 25, 'price': 66.44, 'K': 0.0638, 'level': 1},
            2: {'name': '30mm厚度不燃型复合膨胀聚苯乙烯保温板 ( A 级)', 'thickness': 30, 'price': 69.03, 'K': 0.0624, 'level': 1},
            2: {'name': '35mm厚度不燃型复合膨胀聚苯乙烯保温板 ( A 级)', 'thickness': 35, 'price': 71.62, 'K': 0.0609, 'level': 1},
            3: {'name': '40mm厚度不燃型复合膨胀聚苯乙烯保温板 ( A 级)', 'thickness': 40, 'price': 74.21, 'K': 0.0595, 'level': 1},
            4: {'name': '50mm厚度不燃型复合膨胀聚苯乙烯保温板 ( A 级)', 'thickness': 50, 'price': 79.39, 'K': 0.0566, 'level': 1},
            5: {'name': '60mm厚度不燃型复合膨胀聚苯乙烯保温板 ( A 级)', 'thickness': 60, 'price': 84.57, 'K': 0.0538, 'level': 1},
            6: {'name': '80mm厚度不燃型复合膨胀聚苯乙烯保温板 ( A 级)', 'thickness': 80, 'price': 94.93, 'K': 0.049, 'level': 1},
            7: {'name': '100mm厚度不燃型复合膨胀聚苯乙烯保温板 ( A 级)', 'thickness': 100, 'price': 105.29, 'K': 0.0443, 'level': 1}
        }

class WindowFrameMaterialWarehouse(MaterialWarehouse):
    def __init__(self):
        super().__init__()
        self.load_materials(self.load_window_frame_materials())

        self.pr = PolyRegressor()

    def load_window_frame_materials(self):

        return {
            0: {'type': 'WF-001', 'material': 'L01', 'area': 36.0, 'width': 4.0, 'height': 3.0, 'frame_area': 2.4, 'price': 120, 'window_frame_area_ratio':0.2,'K': 3.2},
            1: {'type': 'WF-001', 'material': 'L01', 'area': 30.0, 'width': 4.0, 'height': 3.0, 'frame_area': 2.4, 'price': 120, 'window_frame_area_ratio':0.2,'K': 3.2},
            2: {'type': 'WF-002', 'material': 'L02', 'area': 28.0, 'width': 4, 'height': 2, 'frame_area': 1.8, 'price': 120, 'window_frame_area_ratio':0.25,'K': 3.2},
            3: {'type': 'WF-003', 'material': 'L03', 'area': 26.0, 'width': 3, 'height': 1, 'frame_area': 0.68, 'price': 120, 'window_frame_area_ratio':0.3,'K': 3.2},
            4: {'type': 'WF-001', 'material': 'L01', 'area': 25.0, 'width': 4.0, 'height': 3.0, 'frame_area': 2.4, 'price': 120, 'window_frame_area_ratio':0.2,'K': 3.2},
            5: {'type': 'WF-001', 'material': 'L01', 'area': 22.0, 'width': 4.0, 'height': 3.0, 'frame_area': 2.4, 'price': 120, 'window_frame_area_ratio':0.2,'K': 3.2},
            6: {'type': 'WF-002', 'material': 'L02', 'area': 20.0, 'width': 4, 'height': 2, 'frame_area': 1.8, 'price': 120, 'window_frame_area_ratio':0.25,'K': 3.2},
            7: {'type': 'WF-003', 'material': 'L03', 'area': 18.0, 'width': 3, 'height': 1, 'frame_area': 0.68, 'price': 120, 'window_frame_area_ratio':0.3,'K': 3.2},
            8: {'type': 'WF-002', 'material': 'L02', 'area': 16.0, 'width': 4, 'height': 2, 'frame_area': 1.8, 'price': 120, 'window_frame_area_ratio':0.25,'K': 3.2},
            9: {'type': 'WF-003', 'material': 'L03', 'area': 12.0, 'width': 3, 'height': 1, 'frame_area': 0.68, 'price': 60, 'window_frame_area_ratio':0.3,'K': 5.9},
            10: {'type': 'WF-001', 'material': 'L01', 'area': 10.0, 'width': 4.0, 'height': 3.0, 'frame_area': 2.4, 'price': 60, 'window_frame_area_ratio':0.2,'K': 5.9},
            11: {'type': 'WF-002', 'material': 'L02', 'area': 7.0, 'width': 4, 'height': 2, 'frame_area': 1.8, 'price': 60, 'window_frame_area_ratio':0.25,'K': 5.9}

        }
        '''
