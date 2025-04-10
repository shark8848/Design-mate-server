from scipy.optimize import brent, fmin, minimize
import numpy as np
import sys
from nameko.rpc import rpc
sys.path.append("..")
from apocolib.SQLiteTools import Sqlite as sqliteTool
#from apocolib.apocolog4p import apoLogger as apolog

np.set_printoptions(suppress=True)
#pd.set_option('display.float_format', lambda x: '%.2f' % x)

'''
玻璃 ：
    x0：面积（m**2）
    x1: 价格（rmb/m**2）
    x2：玻璃热导系数
    x1 = n1/x2（价格/k比，n1为经验常数）
外墙材料 ：
    x3：面积（m**2）
    x4: 价格（rmb/m**2）
    x5：外墙材料热导系数
    x4 = n2/x5（价格/k比，n2为经验常数）
方程式：
求解：
    min f(x0,x1,x2,x3,x4,x5) = x0*x1 + x3*x4
条件等式：
    x0 + x3 = 40 (假定外墙总面积，含窗，窗=玻璃，不考虑缝隙，窗框)
其它条件和条件不等式：
    x0 >= 2.8 (>= 40*7%)
    x1 = 80/x2
    0.95 < x2 < 4.5
    x4 = 65/x5
    0.7 < x5 < 0.95
    1.5 < (x0*x2 + x3*x5)/40 < 3.5
    x0,x1,x2,x3,x4,x5 > 0 
'''
# 约束非线性规划问题(Scipy.optimize.minimize)
class optimizeService(object):
    name = "optimize_service"

    def objFunction(x):  # 定义目标函数
        a, b= 1, 1
        fx = a*x[0]*x[1]+ b*x[3]*x[4]
        #fx = (a*x[0]*x[1]+ b*x[3]*x[4])/1000
        return fx
    # 定义约束条件函数
    def constraint1(x):  # x0 + x3 = 40
        return x[0] + x[3] -40
    def constraint2(x): # x1 = 80/k1
        return x[1]*x[2] - 80
    def constraint3(x): # 1.5 < (x0*x2 + x3*x5)/40 < 3.5
        return (x[0]*x[2] + x[3]*x[5])/40 - 1.5
    def constraint4(x):
        return -((x[0]*x[2]+x[3]*x[5])/40 - 3.5)
    def constraint5(x):
        return x[3]*x[4] -65

    # 定义边界约束

    x0_bnd = (2.8,40)
    x1_bnd = (75,None)
    x2_bnd = (1.5,3.5)
    x3_bnd = (0,40)
    x4_bnd = (0.0,None)
    x5_bnd = (0.7,0.95)
    bnds =(x0_bnd,x1_bnd,x2_bnd,x3_bnd,x4_bnd,x5_bnd)

    # 定义约束条件

    con1 = {'type':'eq','fun':constraint1}
    con2 = {'type':'eq','fun':constraint2}
    con3 = {'type':'ineq','fun':constraint3}
    con4 = {'type':'ineq','fun':constraint4}
    con5 = {'type':'eq','fun':constraint5}

    cons = ([con1,con2,con3,con4,con5])

    x0 = np.array([5,120,1.5,35,80,0.85]) # 定义求解初值
    res = minimize(objFunction, x0, method='SLSQP', bounds=bnds, constraints=cons)
    @rpc
    def optimize_service(self):
    # 求解优化问题
        print("Optimization problem (res):\t{}".format(self.res.message))  # 优化是否成功
        #print("xOpt = {}".format(res.x))  # 自变量的优化值
        #print("min f(x) = {:.4f}".format(res.fun))  # 目标函数的优化值
        x = self.res.x
        fun = self.res.fun
        print("玻璃面积 " + str(round(x[0],4)))
        print("玻璃价格 " + str(round(x[1],2)))
        print("玻璃k值 " + str(round(x[2],4)))
        print("外墙材料面积 " + str(round(x[3],4)))
        print("外墙材料价格 " + str(round(x[4],2)))
        print("外墙材料k值 " + str(round(x[5],4)))
        print("最低成本（元）" + str(round(fun,2)))
        return str(self.res)
