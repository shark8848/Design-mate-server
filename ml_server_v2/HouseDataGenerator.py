import numpy as np
import dataSetBaseParamters as ds_bp
import json
import argparse
import datetime
import pdb
import os
import random

#pdb.set_trace()

class HouseDataGenerator:
    def __init__(self, rooms):
        if not rooms:
            raise ValueError("rooms 参数不能为空")
        self.rooms = rooms

    # 生成指定数量的墙面，并为每个墙面分配一个面积和方向,(外墙)
    def generate_wall_areas(self, total_area, num_walls, r_length, r_width, r_height):

        if num_walls < 1:
            return []

        breakpoints = sorted([0] + list(np.random.rand(num_walls - 1)) + [1])
        differences = [breakpoints[i+1] - breakpoints[i]
                    for i in range(len(breakpoints) - 1)]

        walls = []
        orientations = ds_bp.get_orientation_keys()
        for i, diff in enumerate(differences):
            #wall_area = round(total_area * diff, 4)
            # East and West walls use r_width, South and North walls use r_length
            orientation = random.choice(orientations)

            wall_width = r_width if orientation in [0, 2] else r_length

            # 生成的墙中 80%的概率为完全为外墙，也存在部分外墙可能性
            exterior_ratio = np.random.rand()
            if exterior_ratio < 0.8:
                exterior_ratio = 1  

            exterior_wall_area = round(wall_width * exterior_ratio * r_height,4)

            walls.append({
                "orientation": orientation,
                "area": exterior_wall_area,
                "wall_width": round(wall_width*exterior_ratio,4),
                "wall_height": round(r_height,4),
                "insulation_material": ds_bp.get_random_WIM_one_v2(),
                "window": None
            })

        return walls

    # 为每个墙面生成窗户，根据墙面面积分配窗户面积
    def generate_windows(self, walls, min_window_wall_ratio=0.2, max_window_wall_ratio=0.6):
        #pdb.set_trace()
        for wall in walls:
            # 以 50% 的概率为墙面分配一个窗户
            window_wall_ratio = 0.0
            if np.random.rand() < 0.5:

                # 根据朝向，查找window_wall_ratio_limit
                max_window_wall_ratio = ds_bp.get_window_wall_ratio_limit(wall["orientation"])
                if max_window_wall_ratio <= min_window_wall_ratio:
                    min_window_wall_ratio = min_window_wall_ratio - 0.01


                window_wall_ratio = np.random.uniform(
                    min_window_wall_ratio, max_window_wall_ratio)
                window_area = round(wall["area"] * window_wall_ratio, 4)

                # 根据窗户的面积和宽高比约束来计算窗户的宽度和高度
                # 如果窗户的宽度或高度超过了墙的宽度或高度，则按比例缩小窗户的尺寸,
                # 保证窗户不超过墙的尺寸,保证窗户的宽高比在0.618~1.618之间
                aspect_ratio = np.random.uniform(0.618, 1.618)
                window_width = np.sqrt(window_area * aspect_ratio)
                window_height = window_area / window_width

                if window_width > wall["wall_width"]:
                    window_width = wall["wall_width"]
                    window_height = window_area / window_width
                if window_height > wall["wall_height"]:
                    window_height = wall["wall_height"]
                    window_width = window_area / window_height

                # 为窗户分配玻璃和框架材料
                wfa_ratio,  wf_material = ds_bp.get_best_wf(window_width, window_height)
                #wfa_ratio,  wf_price,wf_k,wf_material = ds_bp.get_best_wf(window_width, window_height)


                wall["window"] = ({
                    "area": round(window_area,4),#窗户面积
                    "window_wall_ratio": round(window_wall_ratio,4), #window/wall
                    "window_width": round(window_width,4), #窗户宽度
                    "window_height": round(window_height,4),#窗户高度
                    "orientation": wall["orientation"],#窗户朝向
                    "glass_material": ds_bp.get_random_GM_one_v2(),#玻璃材料
                    "wfa_ratio": round(wfa_ratio,4),#窗框面积占比
                    "glass_area": round(window_area * (1-wfa_ratio),4),#玻璃面积
                    "wf_material": wf_material, # 窗框材料
                    "wf_area": round(window_area * wfa_ratio,4)#窗框面积
                    #"wf_price": round(wf_price,4),#窗框价格
                    #"wf_k": round(wf_k,4)#窗框热传导系数
                })

        return walls

    # 根据指定的房间面积生成一个房间的相关数据
    def generate_room_data(self,room_type, room_area, 
            min_area_ratio=0.15, max_area_ratio=0.45, min_walls=1, max_walls=3, min_window_wall_ratio=0.3, max_window_wall_ratio=0.6):

        #print("min_area_ratio=",min_area_ratio," max_area_ratio=",max_area_ratio,"min_window_wall_ratio=",min_window_wall_ratio,"max_window_wall_ratio=",max_window_wall_ratio)
        num_walls = np.random.randint(min_walls, max_walls + 1)

        r_area = round(room_area, 4)
        r_length, r_width = self.generate_room_wh(r_area)
        r_height = ds_bp.get_standard_floor_height()

        if num_walls < 1:
            room_data = {
                'room_type': room_type,
                'room_area': round(r_area,4),
                'room_length': round(r_length,4),  # 5.19 新增
                'room_width': round(r_width,4), # 5.19 新增
                'room_height': round(r_height,4), # 5.19 新增
                'total_wall_area': 0,
                'total_window_area': 0,
                'walls': [],
                'total_cost_im': 0,
                'total_cost_g': 0,
                'total_cost_wf': 0,
                'total_cost': 0,
                'total_avg_k': 0
            }
            return room_data

        total_wall_area = room_area * \
            np.random.uniform(min_area_ratio, max_area_ratio)

        walls = self.generate_wall_areas(total_wall_area, num_walls,r_length, r_width,r_height)

        total_window_area = 0
        total_wf_area = 0
        total_exterior_wall_area = 0
        valid_windows = False
#        pdb.set_trace()
        invalid = 0
        while not valid_windows:

            walls = self.generate_windows(walls,min_window_wall_ratio=min_window_wall_ratio,max_window_wall_ratio=max_window_wall_ratio)

            total_window_area = sum( w["window"]["area"] for w in walls if w["window"] is not None)
            #total_wall_area_including_windows = sum(
            total_exterior_wall_area = sum( w["area"]  for w in walls)
            window_wall_ratio = total_window_area / total_exterior_wall_area
                #w["area"] + (w["window"]["area"] if w["window"] is not None else 0) for w in walls)
            #window_wall_ratio = total_window_area / total_wall_area_including_windows

            #print("window_wall_ratio ,",round(window_wall_ratio,4) ,"swwr ",min_window_wall_ratio ," < ",max_window_wall_ratio)
    
            if min_window_wall_ratio <= window_wall_ratio <= max_window_wall_ratio:
                valid_windows = True
            else:
                invalid = invalid + 1

            if invalid > 500:
                raise ValueError("Invalid wwr too times")

        # 外墙保温材料成本
        total_cost_im = sum(w["area"] * w["insulation_material"]["price"]
                        for w in walls)
        # 窗框材料成本 
        #total_cost_wf = sum(w["window"]["wf_area"] * w["window"]["wf_price"]
        total_cost_wf = sum(w["window"]["wf_area"] * w["window"]["wf_material"]["price"]
                        for w in walls if w["window"] is not None)
        # 玻璃材料成本
        total_cost_g = sum(w["window"]["glass_area"] * w["window"]["glass_material"]["price"]
                        for w in walls if w["window"] is not None)

        # 总成本
        total_cost = total_cost_im + total_cost_g + total_cost_wf

        # 平均k值 

        total_avg_k = sum((w["area"] - (w["window"]["area"] if w["window"] is not None else 0)) * w["insulation_material"]["K"] 
                  + (w["window"]["glass_area"] * w["window"]["glass_material"]["K"] if w["window"] is not None else 0)
                  #+ (w["window"]["wf_area"] * w["window"]["wf_k"] if w["window"] is not None else 0)
                  + (w["window"]["wf_area"] * w["window"]["wf_material"]["K"] if w["window"] is not None else 0)
                  for w in walls) / total_exterior_wall_area
        
        room_data = {
            'room_type': room_type,
            'room_area': round(room_area, 4),
            'room_length': round(r_length, 4), 
            'room_width': round(r_width, 4),
            'room_height': round(r_height, 4),
            'total_wall_area': round(total_exterior_wall_area,4),
            'total_window_area': round(total_window_area,4),
            'walls': walls,
            'total_cost_im': round(total_cost_im,2),
            'total_cost_g': round(total_cost_g,2),
            'total_cost_wf': round(total_cost_wf,2), # 新增窗框成本
            'total_cost': round(total_cost,2),
            'total_avg_k': round(total_avg_k,4)
        }

        return room_data

    # 生成一套房子的数据
    def generate_house_data(self,data_dir=None):
        if not self.rooms:
            raise ValueError("rooms 参数不能为空")

        data = []

        directory = f"./houses_json/{data_dir}"
        if not os.path.exists(directory):
            os.makedirs(directory)

        for room in self.rooms:
            (min_area, max_area, min_wall_num, max_wall_num,
            required, min_room_num, max_room_num, min_window_num, max_window_num) = ds_bp.get_room_encoding_info(room)

            room_area = np.random.uniform(min_area, max_area)
            min_area_ratio, max_area_ratio = ds_bp.get_wall_area_ratio_range()
            #min_window_wall_ratio, max_window_wall_ratio = ds_bp.get_window_wall_ratio_range()
            min_window_wall_ratio, max_window_wall_ratio = ds_bp.get_room_wwr_range(room)# 5.19 改为独立获取每个房间的窗墙比
            room_data = self.generate_room_data(room,
                                            room_area,
                                            min_area_ratio=min_area_ratio,
                                            max_area_ratio=max_area_ratio,
                                            min_walls=min_wall_num,
                                            max_walls=max_wall_num,
                                            min_window_wall_ratio=min_window_wall_ratio,
                                            max_window_wall_ratio=max_window_wall_ratio
                                            )
            data.append(room_data)

        # 将生成的数据转换为 JSON 格式
        data_json = {

            "rooms": [
                {
                    "room_type": room_data["room_type"],
                    "room_area": room_data["room_area"],
                    "room_length": room_data["room_length"], # 5.19 新增
                    "room_width": room_data["room_width"], # 5.19 新增
                    "room_height": room_data["room_height"],# 5.19 新增

                    "total_wall_area": room_data["total_wall_area"],
                    "total_window_area": room_data["total_window_area"],
                    "walls": room_data["walls"],
                    "total_cost_im": room_data["total_cost_im"],
                    "total_cost_g": room_data["total_cost_g"],
                    "total_cost_wf": room_data["total_cost_wf"],
                    "total_cost": room_data["total_cost"],
                    "total_avg_k": room_data["total_avg_k"],
                }
                for room_data in data
            ],
            "h_total_cost": round(sum(room_data["total_cost"] for room_data in data),2),
            "h_total_avg_k": round(sum(room_data["total_avg_k"] * room_data["total_wall_area"] for room_data in data) / sum(room_data["total_wall_area"] for room_data in data),4)
        }

        #current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")[:21]
        file_path = f"{directory}/tds_{current_time}.json"
        self.save_house_data_to_json(data_json, file_path)

        return data_json

    def print_house_data(self,data_json):
        rooms = data_json["rooms"]

        for i, room_data in enumerate(rooms):

            print(f"房间 {i+1}:")
            print("  房间类型:", room_data["room_type"])
            print("  房间面积:", room_data["room_area"], "平方米")
            print("  房间长:", room_data["room_length"], "米") # 5.19 新增
            print("  房间宽:", room_data["room_width"], "米") # 5.19 新增
            print("  房间高:", room_data["room_height"], "米") # 5.19 新增
            print("  外墙墙总面积:", room_data["total_wall_area"], "平方米")
            print("  窗户总面积:", room_data["total_window_area"], "平方米")
            print("  墙面信息:")

            for wall in room_data["walls"]:
                print("    - 方向:", wall["orientation"])
                print("      面积:", wall["area"], "平方米")
                print("      墙宽:", wall["wall_width"], "米") # 5.19 新增
                print("      墙高:", wall["wall_height"], "米") # 5.19 新增

                print("      绝热材料:", wall["insulation_material"])
                if wall["window"]:
                    print("      窗户信息:")
                    print("        面积(洞口):", wall["window"]["area"], "平方米")
                    print("        窗墙比:", wall["window"]["window_wall_ratio"]*100, "%") # 6.2 新增
                    print("        窗宽(洞口):", wall["window"]["window_width"], "米") # 5.19 新增
                    print("        窗高(洞口):", wall["window"]["window_height"], "米") # 5.19 新增
                    print("        方向:", wall["window"]["orientation"])
                    print("        玻璃材料:", wall["window"]["glass_material"])
                    print("        窗框比:", wall["window"]["wfa_ratio"]*100, "%") # 5.22 新增
                    print("        玻璃面积:", wall["window"]["glass_area"], "平方米")# 5.22 新增
                    print("        窗框材料:", wall["window"]["wf_material"])# 6.4 新增
                    print("        窗框面积:", wall["window"]["wf_area"], "平方米")# 5.22 新增

                else:
                    print("      无窗户")

            print("  绝热材料总成本:", room_data["total_cost_im"], "元")
            print("  玻璃材料总成本:", room_data["total_cost_g"], "元")
            print("  窗框材料总成本:", room_data["total_cost_wf"], "元") # 新增窗框成本
            print("  总成本:", room_data["total_cost"], "元")
            print("  平均热传导系数:", room_data["total_avg_k"], "W/(m^2*K)")
            print()
        print("总成本:", data_json["h_total_cost"], "元")
        print("平均热传导系数:", data_json["h_total_avg_k"], "W/(m^2*K)")

    # 根据room 的面积 生成长和宽的数据，长宽比为0.5- 黄金分割点
    def generate_room_wh(self, room_area):
        ratio = np.random.uniform(0.618, 1.618)
        width = (room_area * ratio)**0.5
        height = room_area / width
        return width, height
    

    @staticmethod
    # 使用自定义的JSON编码器
    def save_house_data_to_json(house_data, file_path):
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(house_data, json_file, ensure_ascii=False, indent=4, cls=CustomJSONEncoder)
        #print(' saved file ',file_path)

    def ConvertHouseToArray(self, data_json):

        max_num_rooms = ds_bp.get_max_num_rooms()
        max_num_walls = ds_bp.get_max_num_walls()
        max_num_windows = ds_bp.get_max_num_windows()
        wall_material_len = ds_bp.get_WIM_num()
        glass_type_len = ds_bp.get_GM_num()
        wf_material_len = ds_bp.get_WF_num()

        room_feature_len = (
            4 + 2 * max_num_walls + 2 * max_num_windows + wall_material_len + glass_type_len + wf_material_len
        )

        house = data_json
        
        # 从 house 数据中提取房子的特征和目标值
        house_features = [0] * max_num_rooms * room_feature_len
        house_targets = [0] * max_num_rooms * 2

        for room_index, room in enumerate(house["rooms"]):
            room_start = room_index * room_feature_len

            # 添加房间特征
            house_features[room_start] = room['room_type']
            house_features[room_start + 1] = room['room_area']
            house_features[room_start + 2] = room['total_wall_area']
            house_features[room_start + 3] = room['total_window_area']

            for wall_index, wall in enumerate(room['walls']):
                house_features[room_start + 4 + wall_index] = wall['area']
                house_features[room_start + 4 + max_num_walls + wall_index] = wall['orientation']
                #print("house_features 1",len(house_features))

                if wall['window'] is not None:
                    window_index = wall['window']['orientation']
                    house_features[room_start + 4 + 2 * max_num_walls + window_index] = wall['window']['area']
                    house_features[room_start + 4 + 2 * max_num_walls + max_num_windows + window_index] = wall['window']['orientation']
                    #print("house_features 2",len(house_features))

            # 为墙壁保温材料创建一个独热编码向量
            wall_material_encoding = [0] * wall_material_len
            if room['walls']:
                wall_material_key = room['walls'][0]['insulation_material']['key']
                wall_material_encoding[wall_material_key] = 1
            house_features[room_start + 4 + 2 * max_num_walls + 2 * max_num_windows: room_start +
                        4 + 2 * max_num_walls + 2 * max_num_windows + wall_material_len] = wall_material_encoding
            
            # 为玻璃材料创建一个独热编码向量
            glass_material_encoding = [0] * glass_type_len

            if room['walls'] and room['walls'][0]['window']:
                glass_material_key = room['walls'][0]['window']['glass_material']['key']
                glass_material_encoding[glass_material_key] = 1

            start = room_start + 4 + 2 * max_num_walls + 2 * max_num_windows + wall_material_len
            end = room_start + 4 + 2 * max_num_walls + 2 * max_num_windows + wall_material_len + glass_type_len
            house_features[start:end] = glass_material_encoding
            #print("house_features 4",len(house_features))


            # 为窗框材料创建一个独热编码向量 2023.06.4 sunhy
            wf_material_encoding = [0] * wf_material_len

            if room['walls'] and room['walls'][0]['window']:
                wf_material_key = room['walls'][0]['window']['wf_material']['key']
                wf_material_encoding[wf_material_key] = 1

            start = room_start + 4 + 2 * max_num_walls + 2 * max_num_windows + wall_material_len + glass_type_len
            end = room_start + 4 + 2 * max_num_walls + 2 * max_num_windows + wall_material_len + glass_type_len + wf_material_len
            house_features[start:end] = wf_material_encoding
            # ---------------------> end sunhy .2023.06.04
            
            # 添加房间目标值
            house_targets[room_index * 2] = room['total_cost']
            house_targets[room_index * 2 + 1] = room['total_avg_k']

        # 添加房子的总目标值
        house_targets.extend([house['h_total_cost'], house['h_total_avg_k']])

        #print("house_features length ",len(house_features),"house_targets length ",len(house_targets))
        return house_features, house_targets

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)
    
def main(file_path="house_data.json"):
    
    house_generator = HouseDataGenerator(ds_bp.get_random_rooms())
    house_data = house_generator.generate_house_data()
    house_generator.print_house_data(house_data)
    HouseDataGenerator.save_house_data_to_json(house_data, file_path)

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description='Generate and save house data.')
    #parser.add_argument('file_path', type=str, help='The path to save the house data JSON file.')
    #args = parser.parse_args()

    #main(args.file_path)
    #pdb.set_trace()
    main()
