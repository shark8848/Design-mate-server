import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.optimize import minimize
from ThermalPowerCalculator import ThermalPowerCalculator

np.set_printoptions(suppress=True)

# 定义神经网络模型
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 7)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#----------------------------------

area = 120
wall_area = 4 * 3 * 4.5
roof_area = 4.5 * 3
floor_area = 4.5 * 3

window_data = [
#    {"area": 2, "u_value": 2.3, "material": "glass"},
#    {"area": 1, "u_value": 3.5, "material": "wood"}
]

tp_c = ThermalPowerCalculator(area, wall_area, roof_area, floor_area, window_data)
tp_c.set_u_values = {"wall": 1.5, "roof": 1.2, "floor": 1.0}
tp_c.set_temperature_difference = 8
'''
cooling_load = tp_c.cooling_load()
heating_load = tp_c.heating_load()
average_heating_load = tp_c.average_heating_load()
average_cooling_load = tp_c.average_cooling_load()
print(f"area {area} wall_area {wall_area} roof_area {roof_area} floor_area {floor_area}")
print("Cooling Load: {:.2f} kW".format(cooling_load))
print("Heating Load: {:.2f} kW".format(heating_load))
print("average_heating_load: {:.2f} kW/m2".format(average_heating_load))
print("average_cooling_load: {:.2f} kW/m2".format(average_cooling_load))
'''
# 定义成本函数
def cost(x, net):
    x_tensor = torch.Tensor(x).unsqueeze(0)
    output = net(x_tensor)
    Sg, Pg, Sim, Pim, Kg, Kim, im_max_thickness = output.squeeze().tolist()
    return Sg*Pg + Sim*Pim

# 定义约束条件
def constraint(x):
    Sg, Pg, Sim, Pim, Kg, Kim = x
    Sa = Sg + Sim
    K = (Sg*Kg + Sim*Kim)/Sa
    return [
        Sa - 50,  # 外墙总面积为50
        0.4*Sa - Sg,  # 玻璃窗户面积比例大于0.4
        Sg - 0.7*Sa,  # 玻璃窗户面积比例小于0.7
        0.9 - K,  # 整体K值大于0.9
        K - 2.0  # 整体K值小于2.0
        #0 <= im_max_thickness <= 10  # 保温材料厚度在一定范围内
    ]

# 定义变量范围
bounds = [(0, 50), (0, 100), (0, 50), (0, 100), (0.5, 6), (0.5, 2)]

# 定义训练数据和标签
train_data = torch.Tensor(np.random.rand(100, 6))
train_label = torch.Tensor(np.random.rand(100, 7))

# 定义神经网络模型和优化器
#net = Net(6, 16)
net = Net(6, 16)
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.L1Loss()

# 训练神经网络
for epoch in range(100000):
    optimizer.zero_grad()
    output = net(train_data)
    loss = criterion(output, train_label)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss {loss.item()}")

# 使用神经网络进行预测和优化
result = minimize(lambda x: cost(x, net), [50, 80, 0, 60, 3, 1],
method='SLSQP', bounds=bounds, constraints={'type': 'ineq', 'fun': constraint})
#print(result)
print(f"message :{result['message']} ")
print(f"success :{result['success']} ")
print(f"最小化的目标函数值 fun :{result['fun']} ")
print(f"最小化目标函数时的自变量取值x :{result['x']} ")
print(f"迭代次数 nit :{result['nit']} ")
print(f"目标函数计算次数 nfev :{result['nfev']} ")
print(f"约束函数计算次数 njev :{result['njev']} ")
