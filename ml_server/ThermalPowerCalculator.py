import re
import ast

class ThermalPowerCalculator:

    u_values = {"wall": 1.5, "roof": 1.2, "floor": 1.0}
    temperature_difference = 8

    def set_temperature_difference(self,dif):
        temperature_difference = dif

    def set_u_values(self,u_v):
        #u_values = {"wall": 1.5, "roof": 1.2, "floor": 1.0}
        u_values = replace_params(u_v)

    def replace_params(input_str):
        # 将参数字符串转换成Python字典类型
        params = ast.literal_eval(input_str)
        # 定义正则表达式模式
        pattern = r'(?P<key>wall|roof|floor):\s*(?P<value>\d+(?:\.\d+)?)'
        # 定义替换函数
        def repl(match):
            key = match.group('key')
            value = params[key]
            return f'{key}: {value}'
        # 使用re.sub()函数进行替换
        output_str = re.sub(pattern, repl, input_str)
        return output_str
    

    def __init__(self, area, wall_area, roof_area, floor_area, window_data):
        self.area = area
        self.wall_area = wall_area
        self.roof_area = roof_area
        self.floor_area = floor_area
        self.window_data = window_data
        
    def wall_heat_transfer(self, u_value):
        return self.wall_area * u_value

    def roof_heat_transfer(self, u_value):
        return self.roof_area * u_value

    def floor_heat_transfer(self, u_value):
        return self.floor_area * u_value
    
    def window_heat_transfer(self):
        heat_transfer = 0
        for window in self.window_data:
            area = window["area"]
            u_value = window["u_value"]
            heat_transfer += area * u_value
        return heat_transfer
    
    def total_heat_transfer(self):
        wall_heat_transfer = self.wall_heat_transfer(self.u_values["wall"])
        roof_heat_transfer = self.roof_heat_transfer(self.u_values["roof"])
        floor_heat_transfer = self.floor_heat_transfer(self.u_values["floor"])
        window_heat_transfer = self.window_heat_transfer()
        return wall_heat_transfer + roof_heat_transfer + floor_heat_transfer + window_heat_transfer
    
    def cooling_load(self):
        heat_transfer = self.total_heat_transfer()
        return heat_transfer * self.temperature_difference
    
    def heating_load(self):
        heat_transfer = self.total_heat_transfer()
        return heat_transfer * self.temperature_difference * -1

    def power(self):
        return cooling_load(),heating_load()

    def average_heating_load(self):
        return round(float(self.heating_load()/self.area), 3)


    def average_cooling_load(self):
        return round(float(self.cooling_load()/self.area), 3)

if __name__ == "__main__":

    area = 120
    wall_area = 4 * 3 * 4.5
    roof_area = 4.5 * 3
    floor_area = 4.5 * 3

    window_data = [
        {"area": 2, "u_value": 2.3, "material": "glass"},
        {"area": 1, "u_value": 3.5, "material": "wood"}
    ]

    tp_c = ThermalPowerCalculator(area, wall_area, roof_area, floor_area, window_data)
    tp_c.set_u_values = {"wall": 1.5, "roof": 1.2, "floor": 1.0}
    tp_c.set_temperature_difference = 8

    cooling_load = tp_c.cooling_load()
    heating_load = tp_c.heating_load()
    average_heating_load = tp_c.average_heating_load()
    average_cooling_load = tp_c.average_cooling_load()
    print(f"area {area} wall_area {wall_area} roof_area {roof_area} floor_area {floor_area}")
    print("Cooling Load: {:.2f} kW".format(cooling_load))
    print("Heating Load: {:.2f} kW".format(heating_load))
    print("average_heating_load: {:.2f} kW/m2".format(average_heating_load))
    print("average_cooling_load: {:.2f} kW/m2".format(average_cooling_load))

