import numpy as np
from PolyRegressor import PolyRegressor

# 建筑类型
architecture_type_mapping = {
    0: '住宅',
    1: '写字楼',
    2: '公共建筑'
}

# 对上述字典使用二进制进行独热编码
# 住宅：100
# 写字楼：010
# 公共建筑：001
def get_architecture_type_OHE(architecture_type) -> np.ndarray:
    one_hot_encoding = np.zeros(len(architecture_type_mapping))
    one_hot_encoding[architecture_type] = 1
    return one_hot_encoding

def get_architecture_type_OHL() -> np.ndarray:
    one_hot_list = []
    for i in range(len(architecture_type_mapping)):
        one_hot_list.append(get_architecture_type_OHE(i))
    return one_hot_list


#print(get_architecture_type_OHL())

# 从建筑类型中随机选择一个key
#@staticmethod
def get_random_arch_type():
    return np.random.choice(list(architecture_type_mapping.keys()))

# 查询建筑类型
#@staticmethod
def get_arch_type(architecture_type):
    for key, value in architecture_type_mapping.items():
        if value == architecture_type:
            return key
    return None

# 查询所有建筑类型
#@staticmethod
def get_arch_type_mapping():
    return architecture_type_mapping

# 建筑品质分级
building_quality_classification_mapping = {
    0: '普通',
    1: '品质',
    2: '高级'
}

# 对上述字典使用二进制进行独热编码
# 普通：100
# 品质：010
# 高级：001
def get_building_quality_classification_OHE(building_quality_classification) -> np.ndarray:
    one_hot_encoding = np.zeros(len(building_quality_classification_mapping))
    one_hot_encoding[building_quality_classification] = 1
    return one_hot_encoding

def get_building_quality_classification_OHL() -> np.ndarray:
    one_hot_list = []
    for i in range(len(building_quality_classification_mapping)):
        one_hot_list.append(get_building_quality_classification_OHE(i))
    return one_hot_list

#print(get_building_quality_classification_OHL())

# 从建筑品质分级中随机选择一个key
#@staticmethod
def get_random_BQC_OH():
    return np.random.choice(list(get_building_quality_classification_OHL()))

# 从建筑品质分级中随机选择一个key
#@staticmethod
def get_random_BQC():
    return np.random.choice(list(building_quality_classification_mapping.keys()))

# 查询建筑品质分级
#@staticmethod
def get_BQC(building_quality_classification):
    for key, value in building_quality_classification_mapping.items():
        if value == building_quality_classification:
            return key
    return None

# 查询所有建筑品质分级
#@staticmethod
def get_BQC_mapping():
    return building_quality_classification_mapping


# 楼层序号范围
floor_range = (1, 50)

# 随机生成楼层序号
#@staticmethod
def get_random_floor():
    return np.random.randint(floor_range[0], floor_range[1])

# 查询楼层序号范围
#@staticmethod
def get_floor_range():
    return floor_range[0], floor_range[1]

# 房间数量范围（所有类型）
room_range = (6, 12)

# 查询房间数量范围（所有类型）
#@staticmethod
def get_room_range():
    return room_range[0], room_range[1]

# 朝向
orientation_mapping = {
    0: '东',
    1: '南',
    2: '西',
    3: '北'
}
# 对上述字典使用二进制进行独热编码
# 东：1000
# 南：0100
# 西：0010
# 北：0001

def get_orientation_OHE(orientation) -> np.ndarray:
    one_hot_encoding = np.zeros(len(orientation_mapping))
    one_hot_encoding[orientation] = 1
    return one_hot_encoding

def get_orientation_OHL() -> np.ndarray:
    one_hot_list = []
    for i in range(len(orientation_mapping)):
        one_hot_list.append(get_orientation_OHE(i))
    return one_hot_list

#print(get_orientation_OHL())
# 查询所有朝向的key
#@staticmethod
def get_orientation_keys():
    return list(orientation_mapping.keys())

# 查询朝向数量
#@staticmethod
def get_o_num() -> int:
    return len(orientation_mapping)

# 查询朝向
#@staticmethod
def get_orientation(orientation):
    for key, value in orientation_mapping.items():
        if value == orientation:
            return key
    return None

# 查询所有朝向
#@staticmethod
def get_orientation_mapping():
    return orientation_mapping

# 窗墙比朝向限制（成都地区)
window_wall_ratio_limits = {
    0: 0.35,
    1: 0.35,
    2: 0.35,
    3: 0.40
}
# 在所有房间中，面积最大的，让例外出现在这个房间上。
one_max_ww_ratio = 0.6

def get_window_wall_ratio_limit(direction):
    return window_wall_ratio_limits.get(direction, 0.0)

# 房间编码描述说明
# name: 房间名称
# area_range: 房间面积范围
# wall_range: 房间外墙数量范围
# required: 是否必须
# num_range: 一套房子中允许的房间数量范围
# window_range: 房间窗户数量范围
room_encoding = {
    0: {'name': '客厅',     'area_range': (15, 40), 'wall_range': (1, 3),'wwr_range':(0.2,0.6),   'required':True, 'num_range':(1,2),'window_range':(1,3)},
    1: {'name': '主卧',     'area_range': (15, 40), 'wall_range': (1, 3),'wwr_range':(0.2,0.6),   'required':True, 'num_range':(1,2),'window_range':(1,3)},
    2: {'name': '厨房',     'area_range': (4, 15),  'wall_range': (1, 2),'wwr_range':(0.2,0.6),   'required':True, 'num_range':(1,2),'window_range':(1,2)},
    3: {'name': '主卫',     'area_range': (4, 15),  'wall_range': (1, 2),'wwr_range':(0.2,0.6),   'required':True, 'num_range':(1,2),'window_range':(1,2)},
    4: {'name': '生活阳台', 'area_range': (10, 30), 'wall_range': (0, 3),'wwr_range':(0.2,0.6),   'required':False, 'num_range':(0,1),'window_range':(0,2)},
    5: {'name': '餐厅',     'area_range': (15, 40), 'wall_range': (1, 3),'wwr_range':(0.2,0.6),   'required':True, 'num_range':(1,2),'window_range':(1,2)},
    6: {'name': '次卧',     'area_range': (10, 25), 'wall_range': (1, 3),'wwr_range':(0.2,0.6),   'required':False, 'num_range':(0,2),'window_range':(1,2)},
    7: {'name': '卧室',     'area_range': (10, 25), 'wall_range': (1, 3),'wwr_range':(0.2,0.6),   'required':False, 'num_range':(0,2),'window_range':(1,2)},
    8: {'name': '书房',     'area_range': (10, 25), 'wall_range': (1, 3),'wwr_range':(0.2,0.6),   'required':False, 'num_range':(0,2),'window_range':(1,2)},
    9: {'name': '卫生间',   'area_range': (4, 15),  'wall_range': (1, 2),'wwr_range':(0.2,0.6),   'required':False, 'num_range':(0,2),'window_range':(1,2)},
    10: {'name': '储物间',  'area_range': (4, 15),  'wall_range': (0, 2),'wwr_range':(0.2,0.6),   'required':False, 'num_range':(0,1),'window_range':(0,2)},
    11: {'name': '景观阳台','area_range': (10, 30), 'wall_range': (0, 3),'wwr_range':(0.2,0.6),   'required':False, 'num_range':(0,1),'window_range':(0,2)}
}
'''
room_encoding = {
    0: {'name': '客厅',     'area_range': (15, 40), 'wall_range': (1, 3),'wwr_range':(0.6,1),   'required':True, 'num_range':(1,2),'window_range':(1,3)},
    1: {'name': '主卧',     'area_range': (15, 40), 'wall_range': (1, 3),'wwr_range':(0.6,1),   'required':True, 'num_range':(1,2),'window_range':(1,3)},
    2: {'name': '厨房',     'area_range': (4, 15),  'wall_range': (1, 2),'wwr_range':(0.4,0.8),   'required':True, 'num_range':(1,2),'window_range':(1,2)},
    3: {'name': '主卫',     'area_range': (4, 15),  'wall_range': (1, 2),'wwr_range':(0.2,0.6),   'required':True, 'num_range':(1,2),'window_range':(1,2)},
    4: {'name': '生活阳台', 'area_range': (10, 30), 'wall_range': (0, 3),'wwr_range':(0.6,1),   'required':False, 'num_range':(0,1),'window_range':(0,2)},
    5: {'name': '餐厅',     'area_range': (15, 40), 'wall_range': (1, 3),'wwr_range':(0.6,1),   'required':True, 'num_range':(1,2),'window_range':(1,2)},
    6: {'name': '次卧',     'area_range': (10, 25), 'wall_range': (1, 3),'wwr_range':(0.6,0.8),   'required':False, 'num_range':(0,2),'window_range':(1,2)},
    7: {'name': '卧室',     'area_range': (10, 25), 'wall_range': (1, 3),'wwr_range':(0.4,0.7),   'required':False, 'num_range':(0,2),'window_range':(1,2)},
    8: {'name': '书房',     'area_range': (10, 25), 'wall_range': (1, 3),'wwr_range':(0.2,0.6),   'required':False, 'num_range':(0,2),'window_range':(1,2)},
    9: {'name': '卫生间',   'area_range': (4, 15),  'wall_range': (1, 2),'wwr_range':(0.2,0.6),   'required':False, 'num_range':(0,2),'window_range':(1,2)},
    10: {'name': '储物间',  'area_range': (4, 15),  'wall_range': (0, 2),'wwr_range':(0.0,0.4),   'required':False, 'num_range':(0,1),'window_range':(0,2)},
    11: {'name': '景观阳台','area_range': (10, 30), 'wall_range': (0, 3),'wwr_range':(0.6,1),   'required':False, 'num_range':(0,1),'window_range':(0,2)}
}
'''
def get_room_encoding_OHE(r_e) -> np.ndarray:

    one_hot_encoding = np.zeros(len(room_encoding))
    one_hot_encoding[r_e] = 1
    return one_hot_encoding

def get_room_encoding_OHL() -> np.ndarray:
    one_hot_list = []
    for i in range(len(room_encoding)):
        one_hot_list.append(get_room_encoding_OHE(i))
    return one_hot_list

def get_random_rooms_OHL():
    rooms_list = []
    spaces = [0, 1, 2, 3, 4, 5, 6]
    spaces.extend(np.random.choice([7,9]*3 + [8, 10, 11], size=np.random.randint(0, 4), replace=False))

    # 如何spaces数量大于room_range[1]，则随机删除
    if len(spaces) > room_range[1]:
        raise Exception('房间数量超过最大值')
    for s in spaces:
        rooms_list.append(get_room_encoding_OHE(s))

    return rooms_list

#print(get_random_rooms_OHL())

# 从房间编码中随机选择一个key,(0,1,2,3,4,5,6)为必选项，(7,8,9,10,11)为可选项，其中 (7,9) 最多可以选3个,10,11最多可以选1个
# 总数在 room_range 范围内
#@staticmethod
def get_random_rooms():
    spaces = [0, 1, 2, 3, 4, 5, 6]
    spaces.extend(np.random.choice([7,9]*3 + [8, 10, 11], size=np.random.randint(0, 4), replace=False))

    # 如何spaces数量大于room_range[1]，则随机删除
    if len(spaces) > room_range[1]:
        raise Exception('房间数量超过最大值')
    else:
        return spaces

# 查询房间编码描述说明
#@staticmethod
def get_room_encoding(room_encoding):
    for key, value in room_encoding.items():
        if value == room_encoding:
            return key
    return None

# 查询所有房间编码描述说明
#@staticmethod
def get_room_encoding_mapping():
    return room_encoding

# 根据房间编码获取面积范围、外墙数量范围、是否必须、数量范围
# min_area_range, max_area_range, min_wall_range, max_wall_range, required, min_num_range, max_num_range
#@staticmethod
def get_room_encoding_info(r_c):
    r_en = room_encoding[r_c]

    return (r_en['area_range'][0], r_en['area_range'][1], 
            r_en['wall_range'][0], r_en['wall_range'][1], 
            r_en['required'], 
            r_en['num_range'][0], r_en['num_range'][1],
            r_en['window_range'][0], r_en['window_range'][1])

# 根据房间编码获取房间窗墙比范围
#@staticmethod
def get_room_wwr_range(r_c):
    r_en = room_encoding[r_c]

    return (r_en['wwr_range'][0], r_en['wwr_range'][1])

# 标准层高
standard_floor_height = 3.0

# 查询标准层高
#@staticmethod
def get_standard_floor_height():
    return standard_floor_height

# 外墙与总面积比范围
wall_area_ratio_range = (0.8, 1.4)

# 查询外墙与总面积比范围
#@staticmethod
def get_wall_area_ratio_range():
    return wall_area_ratio_range[0], wall_area_ratio_range[1]

# 窗墙比范围
window_wall_ratio_range = (0.4, 0.7)

# 查询窗墙比范围
#@staticmethod
def get_window_wall_ratio_range():
    return window_wall_ratio_range[0], window_wall_ratio_range[1]

# 随机产生一个窗墙比
#@staticmethod
def get_random_WWR():
    return np.random.uniform(window_wall_ratio_range[0], window_wall_ratio_range[1])

# 房子总面积/房间数量比，即房间平均面积范围
house_room_area_ratio_range = (10, 30)

# 查询房子总面积/房间数量比，即房间平均面积范围
#@staticmethod
def get_avg_area_ratio_range():
    return house_room_area_ratio_range[0], house_room_area_ratio_range[1]

# 外墙保温材料配置，编码，厚度，价格，k值，级别
wall_insulation_material = {
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

# 查询外墙保温材料配置
#@staticmethod
def get_wall_insulation_material():
    return wall_insulation_material
# 查询外墙保温材料数量
#@staticmethod
def get_WIM_num():
    return len(wall_insulation_material)

# 根据编码查材料，厚度，价格，k值，级别
#@staticmethod
def get_wall_insulation_material_info(material_encoding):
    m_en = wall_insulation_material[material_encoding]
    return m_en['thickness'], m_en['price'], m_en['K'], m_en['level']

# 随机获取外墙保温材料编码，厚度，价格，k值，级别
#@staticmethod
def get_random_WIM():
    material_encoding = np.random.randint(0, len(wall_insulation_material))
    _IM = get_wall_insulation_material_info(material_encoding)
    return material_encoding,_IM[0], _IM[1], _IM[2], _IM[3]

# 随机获取外墙保温材料的其中一个元素，以字典形式返回
#@staticmethod
def get_random_WIM_one():
    material_encoding = np.random.randint(0, len(wall_insulation_material))
    return wall_insulation_material[material_encoding]

from collections import OrderedDict

def get_random_WIM_one_v2():
    material_encoding = np.random.randint(0, len(wall_insulation_material))
    material = wall_insulation_material[material_encoding]
    ordered_material = {'key': material_encoding, **material}
    return ordered_material

#print(get_random_WIM_one())

#print(get_random_WIM_one_v2())

# 玻璃材料，编码，描述，单双玻，厚度，镀膜，夹层厚度，价格，k值
glass_material = {

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
glass_material = {
    0: {'descriptions': 0, 'S_D': 1, 'thickness': 5, 'coating': 0, 'hollow_material': 0, 'price': 35, 'K': 5.8},
    1: {'descriptions': 1, 'S_D': 1, 'thickness': 5, 'coating': 1, 'hollow_material': 0, 'price': 45, 'K': 3.8},
    2: {'descriptions': 1, 'S_D': 1, 'thickness': 5, 'coating': 2, 'hollow_material': 0, 'price': 55, 'K': 2.8},
    3: {'descriptions': 1, 'S_D': 1, 'thickness': 5, 'coating': 3, 'hollow_material': 0, 'price': 65, 'K': 1.8},
    4: {'descriptions': 1, 'S_D': 1, 'thickness': 5, 'coating': 4, 'hollow_material': 0, 'price': 75, 'K': 1.2},
    5: {'descriptions': 1, 'S_D': 1, 'thickness': 5, 'coating': 5, 'hollow_material': 0, 'price': 85, 'K': 0.8},
    6: {'descriptions': 1, 'S_D': 1, 'thickness': 5, 'coating': 6, 'hollow_material': 0, 'price': 95, 'K': 0.6},
    7: {'descriptions': 1, 'S_D': 1, 'thickness': 5, 'coating': 7, 'hollow_material': 0, 'price': 105, 'K': 0.4},
    8: {'descriptions': 1, 'S_D': 1, 'thickness': 5, 'coating': 8, 'hollow_material': 0, 'price': 115, 'K': 0.2}
}

    #0: {'descriptions': '单层普通玻璃',               'S_D': 1,'thickness': 5, 'coating': 0, 'hollow_material': 0, 'price': 35, 'K': 5.8},
    1: {'descriptions': '单层LOW-E玻璃',              'S_D': 1, 'thickness': 5, 'coating': 1, 'hollow_material': 0, 'price': 41.04, 'K': 3.57},
    2: {'descriptions': '单层LOW-E玻璃',              'S_D': 1, 'thickness': 6, 'coating': 1, 'hollow_material': 0, 'price': 46, 'K': 3.57}
    3: {'descriptions': '单层LOW-E玻璃',              'S_D': 1, 'thickness': 8, 'coating': 1, 'hollow_material': 0, 'price': 57.17, 'K': 3.57},
    4: {'descriptions': '单层LOW-E玻璃',              'S_D': 1, 'thickness': 12,'coating': 1, 'hollow_material': 0, 'price': 64.58, 'K': 3.20}    
    5: {'descriptions': '5中透光Low-E+12A+5透明',     'S_D': 2, 'thickness': 5, 'coating': 1, 'hollow_material': 12, 'price': 113.06, 'K': 1.8696},
    6: {'descriptions': '6中透光Low-E+12A+6透明',     'S_D': 2, 'thickness': 6, 'coating': 1, 'hollow_material': 12, 'price': 123.82, 'K': 1.8628},
    7: {'descriptions': '8中透光Low-E+12A+8透明',     'S_D': 2, 'thickness': 8, 'coating': 1, 'hollow_material': 12, 'price': 145.32, 'K': 1.8494},
    9: {'descriptions': '5中透光Low-E+12A+5透明钢化', 'S_D': 2, 'thickness': 5, 'coating': 1, 'hollow_material': 12, 'price': 125.44, 'K': 1.8696},
    10: {'descriptions': '6中透光Low-E+12A+6透明钢化', 'S_D': 2, 'thickness': 6, 'coating': 1, 'hollow_material': 12, 'price': 137.98, 'K': 1.8628},
    11: {'descriptions': '8中透光Low-E+12A+8透明钢化', 'S_D': 2, 'thickness': 8, 'coating': 1, 'hollow_material': 12, 'price': 166.56, 'K': 1.8494}

    '''
# 随机获取玻璃材料的其中一个元素，以字典形式返回
#@staticmethod
def get_random_GM_one():
    material_encoding = np.random.randint(0, len(glass_material))
    return glass_material[material_encoding]

def get_random_GM_one_v2():
    material_encoding = np.random.randint(0, len(glass_material))
    material = glass_material[material_encoding]
    ordered_material = {'key': material_encoding, **material}
    return ordered_material

#print(get_random_GM_one_v2())



# 随机获取玻璃材料编码，描述，单双玻，厚度，镀膜，夹层厚度，价格，k值
#@staticmethod
def get_random_GM():
    material_encoding = np.random.randint(0, len(glass_material))
    _IM = get_glass_material_info(material_encoding, 'descriptions', 'S_D', 'thickness', 'coating', 'hollow_material', 'price', 'K')
    return material_encoding, _IM[0], _IM[1], _IM[2], _IM[3], _IM[4], _IM[5], _IM[6]

# 查询玻璃材料参数
#@staticmethod
def get_glass_material():
    return glass_material

# 查询玻璃材料数量
#@staticmethod
def get_GM_num():
    return len(glass_material)

# 根据编码和属性（可以多个） 查玻璃材料的描述，单双玻，厚度，镀膜，夹层厚度，价格，k值
#@staticmethod
def get_glass_material_info(material_encoding, *args):
    m_en = glass_material[material_encoding]
    info = []
    for arg in args:
        info.append(m_en[arg])
    return tuple(info)

# 窗框材料配置，编码，类型，材料，总面积，长，宽，窗框面积，价格，材料k值
window_frame_material = {
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

def get_WF_num():
    return len(window_frame_material)

pr = PolyRegressor()
def get_best_wf(target_width, target_height):
    best_fit = None
    best_distance = float('inf')
    best_key = None
    for window_key in window_frame_material:
        window = window_frame_material[window_key]
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
        0.0,0.0,{}
        #return 0.0, 0.0, 0.0
    else:
        return round(pr.predict(target_width*target_height),4), {"key": best_key, **best_fit }
        #return round(pr.predict(target_width*target_height),4),best_fit['price'], best_fit['K'], {"key": best_key, **best_fit }
        #return round(pr.predict(target_width*target_height),4),best_fit['price'], best_fit['K'],best_key
        #return best_fit['window_frame_area_ratio'], best_fit['price'], best_fit['K']

dataset_size = 10000

# 查询数据集大小
#@staticmethod
def get_dataset_size():
    return dataset_size

# 定义训练集和测试集比例
train_test_ratio = 0.8

# 查询训练集和测试集比例
#@staticmethod
def get_train_test_ratio():
    return train_test_ratio

max_num_rooms = 12
def get_max_num_rooms():
    return max_num_rooms

max_num_walls = max_num_rooms * 4
def get_max_num_walls():
    return max_num_walls

max_num_windows = max_num_rooms * 4
def get_max_num_windows():
    return max_num_windows

#wall_material_len = get_WIM_num()
#glass_type_len = get_GM_num()

# room min_k_value,max_k_value

max_k_room = 3.00
min_k_room = 0.05
max_k_house = 1.85
min_k_house = 0.75
max_avg_cost = 200.0
min_avg_cost = 10.0

def get_max_k_room():
    return max_k_room
def get_min_k_room():
    return min_k_room
def get_max_k_house():
    return max_k_house
def get_min_k_house():
    return min_k_house
def get_max_avg_cost():
    return max_avg_cost
def get_min_avg_cost():
    return min_avg_cost
