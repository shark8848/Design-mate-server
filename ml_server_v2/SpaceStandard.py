import numpy as np
from apocolib.MlLogger import mlLogger as ml_logger



class SpaceStandard:

    def __init__(self):
        self.room_encoding = self.load_room_encoding()

    # 房间数量范围（所有类型）
    room_range = (6, 12)
    # room_range = (12*6, 12*12)

    # 查询房间数量范围（所有类型）
    # @staticmethod
    def get_room_range(self):
        return self.room_range[0], self.room_range[1]

    # 窗墙比朝向限制（成都地区)
    window_wall_ratio_limits = {
        0: 0.35,
        1: 0.45,
        2: 0.35,
        3: 0.40
        #        0: 0.35,
        #        1: 0.35,
        #        2: 0.35,
        #        3: 0.40
    }
    # 在所有房间中，面积最大的，让例外出现在这个房间上。
    one_max_ww_ratio = 0.6

    def get_window_wall_ratio_limit(self, direction):
        return self.window_wall_ratio_limits.get(direction, 0.0)

    # 房间编码描述说明
    # name: 房间名称
    # area_range: 房间面积范围
    # wall_range: 房间外墙数量范围
    # required: 是否必须
    # num_range: 一套房子中允许的房间数量范围
    # window_range: 房间窗户数量范围
    # wo_probability: 房间窗户数量分布概率
    def load_room_encoding(self):
        return {
            0:  {'name': '客厅',     'area_range': (15, 40), 'wall_range': (1, 3), 'wwr_range': (0.35, 0.6),  'wfr_range': (1/6, 1/3), 'required': True, 'num_range': (1, 2), 'window_range': (1, 3), 'wo_probability': [0.35, 0.65, 0.15, 0.25]},
            1:  {'name': '主卧',     'area_range': (15, 40), 'wall_range': (1, 3), 'wwr_range': (0.35, 0.6),  'wfr_range': (1/6, 1/3), 'required': True, 'num_range': (1, 2), 'window_range': (1, 3), 'wo_probability': [0.35, 0.65, 0.10, 0.20]},
            2:  {'name': '厨房',     'area_range': (4, 15),  'wall_range': (1, 2), 'wwr_range': (0.30, 0.45), 'wfr_range': (0.01, 1/3), 'required': True, 'num_range': (1, 2), 'window_range': (1, 2), 'wo_probability': [0.25, 0.15, 0.45, 0.55]},
            3:  {'name': '主卫',     'area_range': (4, 15),  'wall_range': (1, 2), 'wwr_range': (0.25, 0.45), 'wfr_range': (0.01, 1/3), 'required': True, 'num_range': (1, 2), 'window_range': (1, 2), 'wo_probability': [0.35, 0.65, 0.15, 0.25]},
            4:  {'name': '生活阳台', 'area_range': (10, 30), 'wall_range': (0, 3), 'wwr_range': (0.30, 0.45),  'wfr_range': (0.01, 1/3), 'required': False, 'num_range': (0, 1), 'window_range': (0, 2), 'wo_probability': [0.45, 0.45, 0.25, 0.25]},
            5:  {'name': '餐厅',     'area_range': (15, 40), 'wall_range': (1, 3), 'wwr_range': (0.35, 0.45),  'wfr_range': (1/6, 1/3), 'required': True, 'num_range': (1, 2), 'window_range': (1, 2), 'wo_probability': [0.35, 0.65, 0.15, 0.25]},
            6:  {'name': '次卧',     'area_range': (10, 25), 'wall_range': (1, 3), 'wwr_range': (0.35, 0.6),   'wfr_range': (1/6, 1/3), 'required': False, 'num_range': (0, 2), 'window_range': (1, 2), 'wo_probability': [0.35, 0.55, 0.25, 0.25]},
            7:  {'name': '卧室',     'area_range': (10, 25), 'wall_range': (1, 3), 'wwr_range': (0.35, 0.6),   'wfr_range': (1/6, 1/3), 'required': False, 'num_range': (0, 2), 'window_range': (1, 2), 'wo_probability': [0.35, 0.35, 0.25, 0.25]},
            8:  {'name': '书房',     'area_range': (10, 25), 'wall_range': (1, 3), 'wwr_range': (0.35, 0.6),   'wfr_range': (1/6, 1/3), 'required': False, 'num_range': (0, 2), 'window_range': (1, 2), 'wo_probability': [0.45, 0.45, 0.35, 0.35]},
            9:  {'name': '卫生间',   'area_range': (4, 15),  'wall_range': (1, 2), 'wwr_range': (0.2, 0.45),   'wfr_range': (0.01, 1/3), 'required': False, 'num_range': (0, 2), 'window_range': (1, 2), 'wo_probability': [0.35, 0.35, 0.45, 0.45]},
            10: {'name': '储物间',  'area_range': (4, 15),  'wall_range': (0, 2), 'wwr_range': (0.2, 0.45),   'wfr_range': (0.01, 1/3), 'required': False, 'num_range': (0, 1), 'window_range': (0, 2), 'wo_probability': [0.15, 0.15, 0.45, 0.65]},
            11: {'name': '景观阳台','area_range': (10, 30), 'wall_range': (0, 3), 'wwr_range': (0.2, 0.45),   'wfr_range': (0.01, 1/3), 'required': False, 'num_range': (0, 1), 'window_range': (0, 2), 'wo_probability': [0.35, 0.35, 0.15, 0.15]},
            12: {'name': '楼梯间',  'area_range': (12, 20), 'wall_range': (0, 2), 'wwr_range': (0.2, 0.45),   'wfr_range': (0.01, 1/3), 'required': False, 'num_range': (1, 4), 'window_range': (0, 2), 'wo_probability': [0.15, 0.15, 0.35, 0.35]},
            13: {'name': '过道',    'area_range': (12, 20), 'wall_range': (0, 2), 'wwr_range': (0.2, 0.45),   'wfr_range': (0.01, 1/3), 'required': False, 'num_range': (0, 4), 'window_range': (0, 2), 'wo_probability': [0.15, 0.15, 0.35, 0.35]},
        }

    def get_room_name(self, r_e):  # r_e: room encoding
        return self.room_encoding[r_e]['name']

    def get_room_encoding_OHE(self, r_e) -> np.ndarray:

        one_hot_encoding = np.zeros(len(self.room_encoding))
        one_hot_encoding[r_e] = 1
        return one_hot_encoding

    def get_room_encoding_OHL(self) -> np.ndarray:
        one_hot_list = []
        for i in range(len(self.room_encoding)):
            one_hot_list.append(self.get_room_encoding_OHE(i))
        return one_hot_list

    def get_random_rooms_OHL(self):
        rooms_list = []
        spaces = [0, 1, 2, 3, 4, 5, 6]
        spaces.extend(np.random.choice(
            [7, 9]*3 + [8, 10, 11], size=np.random.randint(0, 4), replace=False))

        # 如何spaces数量大于room_range[1]，则随机删除
        if len(spaces) > self.room_range[1]:
            raise Exception('房间数量超过最大值')
        for s in spaces:
            rooms_list.append(self.get_room_encoding_OHE(s))

        return rooms_list

    # print(get_random_rooms_OHL())

    # 从房间编码中随机选择一个key,(0,1,2,3,4,5,6)为必选项，(7,8,9,10,11)为可选项，其中 (7,9) 最多可以选3个,10,11最多可以选1个
    # 总数在 room_range 范围内
    # @staticmethod
    def get_random_rooms(self):
        spaces = [0, 1, 2, 3, 4, 5, 6]
        spaces.extend(np.random.choice(
            [7, 9]*3 + [8, 10, 11], size=np.random.randint(0, 4), replace=False))

        # 如何spaces数量大于room_range[1]，则随机删除
        if len(spaces) > self.room_range[1]:
            raise Exception('房间数量超过最大值')
        else:
            ml_logger.info('房间: {}'.format(spaces))
            return spaces

    # 查询房间编码描述说明
    # @staticmethod
    def get_room_encoding(room_encoding):
        for key, value in room_encoding.items():
            if value == room_encoding:
                return key
        return None

    # 查询所有房间编码描述说明
    # @staticmethod
    def get_room_encoding_mapping(self):
        return self.room_encoding

    # 根据房间编码获取面积范围、外墙数量范围、是否必须、数量范围
    # min_area_range, max_area_range, min_wall_range, max_wall_range, required, min_num_range, max_num_range
    # @staticmethod
    def get_room_encoding_info(self, r_c):

        ml_logger.info('r_c: {}'.format(r_c))
        
        r_en = self.room_encoding[r_c]

        return (r_en['area_range'][0], r_en['area_range'][1],
                r_en['wall_range'][0], r_en['wall_range'][1],
                r_en['required'],
                r_en['num_range'][0], r_en['num_range'][1],
                r_en['window_range'][0], r_en['window_range'][1],
                r_en['wo_probability'])

    # 根据房间编码获取房间窗墙比范围
    # @staticmethod
    def get_room_wwr_range(self, r_c):
        r_en = self.room_encoding[r_c]

        return (r_en['wwr_range'][0], r_en['wwr_range'][1])

    # 根据房间编码获取房间窗地比范围
    def get_room_wfr_range(self, r_c):
        r_en = self.room_encoding[r_c]

        return (r_en['wfr_range'][0], r_en['wfr_range'][1])
    # 标准层高
    standard_floor_height = 2.95

    # 查询标准层高
    # @staticmethod
    def get_standard_floor_height(self):
        return self.standard_floor_height

    # 外墙与总面积比范围
    wall_area_ratio_range = (0.8, 1.4)

    # 查询外墙与总面积比范围
    # @staticmethod
    def get_wall_area_ratio_range(self):
        return self.wall_area_ratio_range[0], self.wall_area_ratio_range[1]

    # 窗墙比范围
    window_wall_ratio_range = (0.35, 0.6)

    # 查询窗墙比范围
    # @staticmethod
    def get_window_wall_ratio_range(self):
        return self.window_wall_ratio_range[0], self.window_wall_ratio_range[1]

    # 随机产生一个窗墙比
    # @staticmethod
    def get_random_WWR(self):
        return np.random.uniform(self.window_wall_ratio_range[0], self.window_wall_ratio_range[1])

    # 房子总面积/房间数量比，即房间平均面积范围
    house_room_area_ratio_range = (10, 30)

    # 查询房子总面积/房间数量比，即房间平均面积范围
    # @staticmethod
    def get_avg_area_ratio_range(self):
        return self.house_room_area_ratio_range[0], self.house_room_area_ratio_range[1]

    dataset_size = 10000

    # 查询数据集大小
    # @staticmethod
    def get_dataset_size(self):
        return self.dataset_size

    # 定义训练集和测试集比例
    train_test_ratio = 0.8

    # 查询训练集和测试集比例
    # @staticmethod
    def get_train_test_ratio(self):
        return self.train_test_ratio

    max_num_rooms = 12

    def get_max_num_rooms(self):
        return self.max_num_rooms

    max_num_walls = max_num_rooms * 4

    def get_max_num_walls(self):
        return self.max_num_walls

    max_num_windows = max_num_rooms * 4

    def get_max_num_windows(self):
        return self.max_num_windows

    # wall_material_len = get_WIM_num()
    # glass_type_len = get_GM_num()

    # room min_k_value,max_k_value

    max_k_room = 3.00
    min_k_room = 0.05
    max_k_house = 1.85
    min_k_house = 0.75
    max_avg_cost = 200.0
    min_avg_cost = 10.0

    def get_max_k_room(self):
        return self.max_k_room

    def get_min_k_room(self):
        return self.min_k_room

    def get_max_k_house(self):
        return self.max_k_house

    def get_min_k_house(self):
        return self.min_k_house

    def get_max_avg_cost(self):
        return self.max_avg_cost

    def get_min_avg_cost(self):
        return self.min_avg_cost
