import numpy as np
import random
import sys
import os
import argparse
import time
from datetime import datetime

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import custom_object_scope
import MagicalDatasetProducer_v2 as mdsp
import dataSetBaseParamters as ds_bp
import tensorflow as tf
import pdb
from AC2NNetTrainer import loss
from AC2NNetTrainer import ConstraintLayer,optimizeOutputClipLayer,BackpropLayer
sys.path.append("..")
from apocolib.MlLogger import mlLogger as ml_logger
from apocolib.RabbitMQProducer import RabbitMQProducer
from apocolib.MQueueManager import MQueueManager
from apocolib import RpcProxyPool
from apocolib.JSONToExcelConverter import JSONToExcelConverter
from dataSetBaseParamters import *
from BuildingSpaceBase import *
from MaterialWarehouse import *


from apocolib.PDFConverter import excel_to_pdf 
#from tensorflow.python.ops.numpy_ops import np_config
#np_config.enable_numpy_behavior()
np.set_printoptions(threshold=np.inf)


rmqp = RabbitMQProducer(queue_manager=MQueueManager())
pool = RpcProxyPool.RpcProxyPool()

class GAHousePredictor:
    """
    使用遗传算法和神经网络模型来预测给定房屋参数的成本和 K 值。
    """
    def __init__(self, num_generations=10, population_size=10, mutation_rate=0.1):
        """
        GAHousePredictor 类的构造函数。
        初始化遗传算法的参数和模型。
        """
        # 遗传算法参数
        self.num_generations = num_generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.min_cost = ds_bp.get_min_avg_cost()
        self.max_cost = ds_bp.get_max_avg_cost()
        self.min_k = ds_bp.get_min_k_house()
        self.max_k = ds_bp.get_max_k_house()
        '''
        self.min_k = ds_bp.get_min_k_house()
        self.max_k = ds_bp.get_max_k_house()
        self.min_cost = ds_bp.get_min_avg_cost()
        self.max_cost = ds_bp.get_max_avg_cost()
        self.max_num_walls = ds_bp.get_max_num_walls()
        self.max_num_windows = ds_bp.get_max_num_windows()
        self.wall_material_len = ds_bp.get_WIM_num()
        self.glass_type_len =   ds_bp.get_GM_num()
        self.wf_material_len = ds_bp.get_WF_num() # added by sunhy 2023.06.04
        self.max_num_rooms = ds_bp.get_max_num_rooms()
        '''

        # 种群的初始化
        self.population = []

        # 模型和缩放器的初始化
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()

        # 加载模型和缩放器

        custom_objects = {'optimizeOutputClipLayer': optimizeOutputClipLayer, 'ConstraintLayer': ConstraintLayer, 'BackpropLayer': BackpropLayer, 'loss': loss}
        self.model = load_model("./net_model/house_model.h5", custom_objects=custom_objects)

        #self.model = load_model("./net_model/house_model.h5", custom_objects={'loss': loss})
        self.x_scaler.scale_ = np.load("./net_model/x_scaler_scale.npy")
        self.x_scaler.min_ = np.load("./net_model/x_scaler_min.npy")
        self.y_scaler.scale_ = np.load("./net_model/y_scaler_scale.npy")
        self.y_scaler.min_ = np.load("./net_model/y_scaler_min.npy")

        # 日志初始化
        self.current_time = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:21] #strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir = f'net_logs/ga_visualization/{self.current_time}'
        os.makedirs(self.log_dir)
        self.writer = tf.summary.create_file_writer(self.log_dir)

    #  fitness 函数，用于评估个体的适应度（即成本）
    def fitness_calculator(self, cost, k, cost_weight=0.5, k_weight=0.5):

        # 归一化处理，使其值处于 0-1之间
        normalized_cost = (cost - self.min_cost) / (self.max_cost - self.min_cost)
        normalized_k = (k - self.min_k) / (self.max_k - self.min_k)

        # 优化目标是成本和K值都尽可能小，所以适应度函数应当随着这两个值的增大而减小。
        fitness = 1/( cost_weight * (1 - normalized_cost) + k_weight * (1 - normalized_k) + 1e-8 )
        return round(fitness,4)

    def fitness(self,individual):

        params = individual

        if not isinstance(params, np.ndarray):
            params = np.array(params)
            params = params.reshape(1, -1)

        #ml_logger.info(f"fitness-params 行数: {params.shape[0]} 列数: {params.shape[1]}")
        #ml_logger.info(f"x_scaler行数: {self.x_scaler.scale_.shape[0]} x_scaler列数: {self.x_scaler.scale_.shape[1]}")
        #ml_logger.info(f"x_scaler的值: {self.x_scaler.scale_} x_scaler的形状: {self.x_scaler.scale_.shape}")


        params_scaled = self.x_scaler.transform(params)
        predicted_costs_k = self.model.predict([params_scaled])[0]
        predicted_costs_k_unscaled = self.y_scaler.inverse_transform([predicted_costs_k])[0]

        print("predicted_costs_k_unscaled \r",predicted_costs_k_unscaled) #," length ",len(predicted_costs_k_unscaled))

        print("Predicted_costs_k:\r")
        rmqp.publish('')
        rmqp.publish("Predicted_costs_k:")

        temp_list = []  # create a temporary list to hold values
        for i, value in enumerate(predicted_costs_k_unscaled):
            formatted_value = format(value, '.10f')
            print(formatted_value, end='\t')
            temp_list.append(formatted_value)  # add value to list instead of publishing it immediately
            if (i + 1) % 4 == 0:
                rmqp.publish(' '.join(temp_list))  # publish all 4 values as a single message
                print()
                temp_list = []  # reset the list after publishing

        #print()

        #room_feature_len = 4 + 2 * self.max_num_walls + 2 * self.max_num_windows + self.wall_material_len + self.glass_type_len + self.wf_material_len # added self.wf_material_len ,sunhy 2023.06.04
        #pdb.set_trace()
        room_feature_len = Room().get_room_features_len()
        total_room_area = 0
        '''        
        room_start = 0
        for room_index in range(Room.ROOM_NUMBER):
            room_start = room_index * room_feature_len
            room_area = params[0][room_start+1] #individual[room_start + 1]
            total_room_area += room_area
            '''
        # 指定数组下标
        indices = np.arange(1, len(params[0]), room_feature_len)
        # 根据指定下标求和
        total_room_area = np.sum(params[0][indices])
        '''
        if total_room_area < 160:
            print("params[0]", params[0], "len ",len(params[0]))
            sys.exit()
            '''

        print("total_room_area :",round(total_room_area,4),"avg_cost :", round(predicted_costs_k_unscaled[-2]/total_room_area,4)," avg_k :" ,round(predicted_costs_k_unscaled[-1],4))
        #ml_logger.info(f"total_room_area :{round(total_room_area,4)} avg_cost :{round(predicted_costs_k_unscaled[-2]/total_room_area,4)} avg_k :{round(predicted_costs_k_unscaled[-1],4)}")

        rmqp.publish(f"total_room_area :{round(total_room_area,4)} avg_cost :{round(predicted_costs_k_unscaled[-2]/total_room_area,4)} avg_k :{round(predicted_costs_k_unscaled[-1],4)}")

        return round(self.fitness_calculator(predicted_costs_k_unscaled[-2]/total_room_area,predicted_costs_k_unscaled[-1],0.9,0.1),4)

    # 选择操作，用于从种群中选择两个个体作为亲代
    def selection(self,population):

        # 从种群中随机选择两个个体
        individuals = random.sample(population, 2)

        # 使用 fitness 函数作为评估标准，选择适应度较高（K值较小）的个体
        return min(individuals, key=self.fitness)

    # 返回参数说明：[要更新位置的值,原位置的值,需要更新的位置,原来的位置]
    def mutate_parameter(self,index, value, feature_type):
        #ml_logger.info(f"feature_type {feature_type} ")
        if feature_type == "window_size":
            mutation_value = random.uniform(-0.01, 0.01)  # 按需调整突变范围
            #new_value = max(0, value + mutation_value)
            new_value = value*(1+mutation_value)
            return new_value,new_value,index,index
        elif feature_type == "wall_material":
            #new_value = random.randint(0, wimWarehouse.get_size() - 1)
            #return 1,0,new_value,index
            return 1,0,self.get_new_index(feature_type,index,wimWarehouse.get_size()),index
        elif feature_type == "glass_material":
            #new_value = random.randint(0, gmWarehouse.get_size() - 1)
            #return 1,0,new_value,index
            return 1,0,self.get_new_index(feature_type,index,gmWarehouse.get_size()),index
        elif feature_type == "wf_material":
            #new_value = random.randint(0, wfmWarehouse.get_size() -1)
            #return 1,0,new_value,index
            return 1,0,self.get_new_index(feature_type,index,wfmWarehouse.get_size()),index
        else:
            raise ValueError("Invalid feature type")

        #return new_value
        return value ,value ,index ,index

    def get_new_index(self,feature_type,index,max_step):

        i = 0
        while i < 10: # 确保新的index 落在同一个feature 的下标范围内
            step = random.randint(0, max_step -1)
            n_step = index + step
            if House().find_feature_name(n_step) == feature_type :
                return n_step
            i = i + 1

        # 否则，返回原下标
        return index


    # 交叉操作，用于生成两个后代
    def crossover(self,parent1, parent2):
        #pdb.set_trace()
        # 随机选择一个交叉点，该点位于两个父代个体的有效索引范围内
        #crossover_point = random.randint(0, len(parent1) - 1)
        #crossover_point = random.randint(1, len(parent1) // 12) * 12  # 交叉点必须是12的倍数
        # 确定交叉点
        num_genes = len(parent1)
        num_segments = num_genes // 12  # 分段数
        crossover_point = random.randint(0, 12-1) * num_segments  # 交叉点必须是12的倍数


        # 使用交叉点将两个父代个体划分为两部分，并将对应部分组合成新的子代个体
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        # 返回生成的两个子代个体
        #ml_logger.info(f"child1 {child1} \r child2 {child2}")
        return child1, child2

    # 变异
    def mutation(self,individual, mutation_rate):
        #pdb.set_trace()
        #mutated_individual = individual.copy()
        mutated_individual = np.concatenate(individual.copy()).flatten()
        #ml_logger.info(f"mutated_individual ,{mutated_individual} len = {len(mutated_individual)}")

        for i in range(len(mutated_individual)):

            r = random.random()
            #ml_logger.info(f"index {i} len {len(mutated_individual)} random.random = {r},mutation_rate = {mutation_rate}")

            if r < mutation_rate:
                feature_type = None
                feature_type = House().find_feature_name(i)
                #print("feature_type  ---",feature_type)
                #ml_logger.info(f"index {i} mutation feature_type {feature_type} mutation_rate: {mutation_rate}")
                # 根据索引确定特征类型

                if feature_type is not None:
                    if feature_type == "No mutation allowed": # 非可变变异数据区
                        continue
                    # 如果窗口面积为0，跳过突变
                    if feature_type == "window_size" and mutated_individual[i] == 0:
                        continue

                    #ml_logger.info(f"mutate ------,{feature_type}")

                    mutated_value, o_value, c_index, o_index = self.mutate_parameter(i, mutated_individual[i], feature_type)
                    mutated_individual[o_index] = o_value # 更新原位置的值
                    mutated_individual[c_index] = mutated_value # 更新新位置的值

        return [mutated_individual]

    def run(self,loadFromFile=None,outFile=None,url=None):
        #pdb.set_trace()
        """
        运行遗传算法的主函数。
        """
        # 种群的初始化
        test_house_x ,test_house_y = mdsp.generate_test_population(self.population_size,loadFromFile)

        out_file = None
        if outFile is None:
            #out_file = mdsp.generate_excel_filename(loadFromFile=loadFromFile)
            out_file = mdsp.generate_excel_filename(loadFromFile)
        else:
            out_file = outFile
        # 将预测对象写入excel表
        j2ec = JSONToExcelConverter(['key'])
        j2ec.json_to_excel(filename=mdsp.generate_cn_filename(loadFromFile), output_filename=out_file, model="FILE", sheet_name="Prediction target")
        
        json_data = None
        with open(loadFromFile, 'r') as f:
            json_data = json.load(f)

        # 将预测对象加载转换为 对象House

        self.population = [[test_house_x[i].copy()] for i in range(self.population_size)]

        best_individual = None

        # 运行遗传算法
        for generation in range(self.num_generations):
            new_population = []

            # 对种群中的一半个体进行遗传操作（因为每次迭代会产生两个后代）
            for _ in range(self.population_size // 2):
                # 从种群中选择两个亲代
                parent1 = self.selection(self.population)
                parent2 = self.selection(self.population)

                # 通过交叉操作产生两个后代
                child1, child2 = self.crossover(parent1, parent2)

                # 以突变率的概率对第一个后代进行突变
                '''
                if random.random() < self.mutation_rate:
                    child1 = self.mutation(child1,0.01)
                    '''
                child1 = self.mutation(child1,self.mutation_rate)

                # 以突变率的概率对第二个后代进行突变
                '''
                if random.random() < self.mutation_rate:
                    child2 = self.mutation(child2,0.01)
                    '''
                child2 = self.mutation(child2,self.mutation_rate)


                # 将产生的两个后代添加到新种群中
                new_population.extend([child1, child2])

            # 使用新种群替换旧种群
            self.population = new_population

            # 找到具有最佳适应度的个体
            best_individual = min(self.population, key=self.fitness)

            best_fitness = self.fitness(best_individual)
            with self.writer.as_default():
                tf.summary.scalar('best_fitness', best_fitness, step=generation)

            # 打印当前代数、最佳个体及其适应度
            print("Generation", generation, 
                  "Fitness:", best_fitness ) #self.fitness(best_individual))
            rmqp.publish(f"Generation: {generation} Fitness:{best_fitness} ") #self.fitness(best_individual)}")
            #ml_logger.info(f"Generation {generation} ,Fitness: {best_fitness}")

        # 关闭 SummaryWriter
        self.writer.close()

        # 使用神经网络进行预测
        # 将最佳个体转换为相应的参数
        best_params = best_individual

        # 对参数进行缩放，以匹配神经网络的输入
        best_params_scaled = self.x_scaler.transform(np.array(best_params).reshape(1, -1))

        # 使用神经网络预测最佳个体的总成本和平均K值
        predicted_costs_k = self.model.predict([best_params_scaled])[0]

        # 将预测结果逆缩放，以获得实际的总成本和平均K值
        predicted_costs_k_unscaled = self.y_scaler.inverse_transform(
            [predicted_costs_k])[0]

        print("-----------------------------------------------------------------")
        print("Best solution:")
        print("  Total cost(Rmb):", round(predicted_costs_k_unscaled[-2],2))
        print("  Total average K(W/(m^2*K)):", round(predicted_costs_k_unscaled[-1],6))
        print("-----------------------------------------------------------------")
        rmqp.publish("-----------------------------------------------------------------")
        rmqp.publish("Generating prediction report ......")
        #rmqp.publish("Best solution:")
        #rmqp.publish(f"  Total cost(Rmb):{round(predicted_costs_k_unscaled[-2],2)}")
        #rmqp.publish(f"  Total average K(W/(m^2*K)):{round(predicted_costs_k_unscaled[-1],6)}")
        rmqp.publish("-----------------------------------------------------------------")
        #rmqp.publish(f"Download url: predicted report {url}")

        # 将预测结果更新到对象中
        print(best_params)

        print(np.array(best_params))
        p_house = mdsp.tensor_to_house(json_data = json_data, house_features = np.array(best_params)[0], targets = predicted_costs_k_unscaled)

        # 将结果转换为 json，追加写入excel 文件中
        p_house_json = p_house.to_json_cn()
        j2ec.json_to_excel(filename=loadFromFile, output_filename=out_file, data=p_house_json, model="DATA", sheet_name="Prediction result")

        # 将两个 sheet 中不一样部分自动标识出来
        j2ec.compare_sheets(out_file,"Prediction target","Prediction result")

        # 生成pdf 文件
        excel_to_pdf(excel_file=out_file,sheet_name="Prediction result",pdf_file=f'{out_file}.pdf',summary_data=p_house.get_cost_view())

        rmqp.publish("Generating prediction report, 1 Excel file and 1 PDF file.")
        rmqp.publish(f"Download url: predicted report {url}") # exl
        rmqp.publish(f"Download url: predicted report {url}.pdf") # pdf

        return True


def main(loadFromFile=None, outFile=None, url=None):
    """
    主函数，创建 GAHousePredictor 类的实例并运行遗传算法。
    """
    start_time = time.time()  # 记录开始时间

    #house_predictor = GAHousePredictor(20,32,0.05)
    #house_predictor = GAHousePredictor(20,50,0.02)
    house_predictor = GAHousePredictor(10, 50, 0.02)
    r = house_predictor.run(loadFromFile, outFile, url)

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算执行时间

    print(f"Execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser(description="Run GA House Predictor")
    parser.add_argument('--loadFromFile', type=str, help='Path to the file to load from')
    args = parser.parse_args()
    main(args.loadFromFile)
    '''
    parser = argparse.ArgumentParser(description="Run GA House Predictor")
    parser.add_argument('--loadFromFile', type=str, help='Path to the file to load from')
    parser.add_argument('--outFile', type=str, help='Path to the file to write')
    parser.add_argument('--url', type=str, help='url to the file to download')
    parser.add_argument('--taskId', type=str, help='task id')

    args = parser.parse_args()
    rmqp.publish("") # rmqp 在第一次publish 时才会申请队列.
    rmqp.publish(f"COMMAND_KEY_WORD:prediction task begins-{args.taskId}")
    queue_name = rmqp.get_queue_name()
    # call rpc ,update queue_name for task_id
    rpc_proxy = pool.get_connection()
    result = rpc_proxy.AC2NNetPredicterService.assign_queue_name(args.taskId,queue_name)
    ml_logger.info("update task_id ",args.taskId,"'s queue_name = ",queue_name)
    pool.put_connection(rpc_proxy)

    main(args.loadFromFile,args.outFile,args.url)
    ml_logger.info("predict :",args.loadFromFile, " out put file :" ,args.outFile, " url :", args.url, "task_id :", args.taskId )

    rmqp.publish(f"COMMAND_KEY_WORD:prediction task completed-{args.taskId}")
    rmqp.close()
