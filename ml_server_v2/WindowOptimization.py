import pulp
import numpy as np
import random


class WindowOptimization:
    def __init__(self, area, height, window_wall_ratio_limits, window_area_ratio_limits):
        self.area = area
        self.golden_ratio = random.uniform(0.618, 1.382)
        self.L, self.W = self.generate_params(area)
        self.H = height
        self.window_wall_ratio_limits = window_wall_ratio_limits
        self.window_area_ratio_limits = window_area_ratio_limits
        self.model = pulp.LpProblem(
            "Maximize_Zero_Window_Areas", pulp.LpMaximize)
        self.Sew = pulp.LpVariable("Sew", lowBound=0, upBound=self.W *
                                   self.H * self.window_wall_ratio_limits[0], cat='Continuous')
        self.Ssw = pulp.LpVariable("Ssw", lowBound=0, upBound=self.L *
                                   self.H * self.window_wall_ratio_limits[1], cat='Continuous')
        self.Sww = pulp.LpVariable("Sww", lowBound=0, upBound=self.W *
                                   self.H * self.window_wall_ratio_limits[2], cat='Continuous')
        self.Snw = pulp.LpVariable("Snw", lowBound=0, upBound=self.L *
                                   self.H * self.window_wall_ratio_limits[3], cat='Continuous')

    def generate_params(self, area):
        if area <= 0:
            raise ValueError("面积必须大于0")
        length = (area / self.golden_ratio) ** 0.5
        width = area / length
        return length, width

    def optimize(self):
        self.model += pulp.lpSum([self.Sew == 0,
                                 self.Ssw == 0, self.Sww == 0, self.Snw == 0])

        self.model += self.Sew >= 0
        self.model += self.Ssw >= 0
        self.model += self.Sww >= 0
        self.model += self.Snw >= 0

        self.model += self.Sew <= self.W * \
            self.H * self.window_wall_ratio_limits[0]
        self.model += self.Ssw <= self.L * \
            self.H * self.window_wall_ratio_limits[1]
        self.model += self.Sww <= self.W * \
            self.H * self.window_wall_ratio_limits[2]
        self.model += self.Snw <= self.L * \
            self.H * self.window_wall_ratio_limits[3]

        self.model += self.Sew + self.Ssw + self.Sww + \
            self.Snw >= self.area * self.window_area_ratio_limits[0]
        self.model += self.Sew + self.Ssw + self.Sww + \
            self.Snw <= self.area * self.window_area_ratio_limits[1]

        

        # self.model.solve()
        # Set solver's msg to 0 to suppress output
        self.model.solve(pulp.PULP_CBC_CMD(msg=0))
        result = [round(self.Sew.varValue, 4), round(self.Ssw.varValue, 4), round(self.Sww.varValue, 4), round(self.Snw.varValue, 4)]
        return result

    def get_LWH(self):
        return [round(self.L, 4), round(self.W, 4), self.H]

    def print_results(self):
        Sew = self.Sew.varValue
        Ssw = self.Ssw.varValue
        Sww = self.Sww.varValue
        Snw = self.Snw.varValue

        window_count = sum(x > 0 for x in [Sew, Ssw, Sww, Snw])

        print(
            "------------------------------------------------------------------------------")
        print(f"房间面积 {self.area} L: {self.L:.2f}, W: {self.W:.2f}")
        print(f"Sew: {Sew:.2f}, Ssw: {Ssw:.2f}, Sww: {Sww:.2f}, Snw: {Snw:.2f}")
        print(f"窗户数量: {window_count} 总窗户面积: {Sew + Ssw + Sww + Snw:.2f} 窗地比: {(Sew + Ssw + Sww + Snw) / (self.L * self.W):.2f}")
        print(f"东墙窗墙比: {Sew / (self.W * self.H) * 100:.2f} % 南墙窗墙比: {Ssw / (self.L * self.H) * 100:.2f} % 西墙窗墙比: {Sww / (self.W * self.H) * 100:.2f} % 北墙窗墙比: {Snw / (self.L * self.H) * 100:.2f} %")


def main():
    window_wall_ratio_limits = [0.35, 0.45, 0.35, 0.40]
    window_area_ratio_limits = [1/6, (1 / 6)*1.4]

    for _ in range(100):
        area = np.random.randint(15, 45)
        window_opt = WindowOptimization(
            area, 2.95, window_wall_ratio_limits, window_area_ratio_limits)
        w_areas = window_opt.optimize()
        LWH = window_opt.get_LWH()
        print("------------------------------------------------------------------------------")
        print(area, LWH, w_areas)
        window_opt.print_results()


if __name__ == "__main__":
    main()

