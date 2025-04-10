import pdb
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tqdm import tqdm
import tensorflow as tf
import pickle
from datetime import datetime
import time
import argparse
import os
import random
import numpy as np
from BuildingElement import *
from apocolib.PDFConverter import excel_to_pdf
from apocolib.CouchDBPool import couchdb_pool
from MaterialWarehouse import *
from BuildingSpaceBase import *
from apocolib.JSONToExcelConverter import JSONToExcelConverter
from apocolib import RpcProxyPool
from apocolib.MQueueManager import MQueueManager
from apocolib.RabbitMQProducer import RabbitMQProducer
from apocolib.MlLogger import mlLogger as ml_logger
from apocolib.timing import timing_decorator
import MagicalDatasetProducer_v2 as mdsp
import dataSetBaseParamters as ds_bp
from AC2NNetTrainer_horovod import loss
from AC2NNetTrainer_horovod import ConstraintLayer, optimizeOutputClipLayer, BackpropLayer
import sys
import zlib
from mpi4py import MPI

sys.path.append("..")


# from dataSetBaseParamters import *


np.set_printoptions(threshold=np.inf)
rmqp = RabbitMQProducer(queue_manager=MQueueManager())
pool = RpcProxyPool.RpcProxyPool()


class GAHousePredictorOnCUDA:

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
        self.room_feature_len = Room().get_room_features_len()
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

        custom_objects = {'optimizeOutputClipLayer': optimizeOutputClipLayer,
                          'ConstraintLayer': ConstraintLayer, 'BackpropLayer': BackpropLayer, 'loss': loss}
        # self.model = load_model("./net_model/house_model.h5", custom_objects=custom_objects) # removed 2023.10.19 sunhy

        # self.model = load_model("./net_model/house_model.h5", custom_objects={'loss': loss})
        self.x_scaler.scale_ = np.load("./net_model/x_scaler_scale.npy")
        self.x_scaler.min_ = np.load("./net_model/x_scaler_min.npy")
        self.y_scaler.scale_ = np.load("./net_model/y_scaler_scale.npy")
        self.y_scaler.min_ = np.load("./net_model/y_scaler_min.npy")

        # 日志初始化
        self.current_time = datetime.now().strftime("%Y%m%d_%H%M%S%f")[
            :21]  # strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir = f'net_logs/ga_visualization/{self.current_time}'
        os.makedirs(self.log_dir)
        self.writer = tf.summary.create_file_writer(self.log_dir)

        self.excel_file_titles = ['building', 'floors',
                                  'houses', 'rooms', 'walls', 'windows', 'args']

    @timing_decorator
    #  fitness 函数，用于评估个体的适应度（即成本）
    def fitness_calculator(self, cost, k, cost_weight=0.5, k_weight=0.5):
        # pdb.set_trace()

        # 归一化处理，使其值处于 0-1之间
        normalized_cost = (cost - self.min_cost) / \
            (self.max_cost - self.min_cost)
        normalized_k = (k - self.min_k) / (self.max_k - self.min_k)

        # 优化目标是成本和K值都尽可能小，所以适应度函数应当随着这两个值的增大而减小。
        fitness = 1/(cost_weight * (1 - normalized_cost) +
                     k_weight * (1 - normalized_k) + 1e-8)
        return round(fitness, 4)

    @timing_decorator
    def fitness(self, individual):
        # pdb.set_trace()

        params = individual

        if not isinstance(params, np.ndarray):
            params = np.array(params)
            params = params.reshape(1, -1)

        params_scaled = self.x_scaler.transform(params)

        # 进行推理,on cuda engine
        rpc_proxy = pool.get_connection()

        input_data = pickle.dumps(params_scaled.astype(np.float32))
        predicted_costs_k = rpc_proxy.tensorRT_Service.infer(
            input_data)  # 调用rpc 前 需序列化
        predicted_costs_k = pickle.loads(predicted_costs_k)  # 对结果反序列化
        pool.put_connection(rpc_proxy)
        predicted_costs_k_unscaled = self.y_scaler.inverse_transform(
            [predicted_costs_k]).astype(np.float32)[0]

        # ml_logger.info(f'predicted_costs_k_unscaled: {predicted_costs_k_unscaled}')

        avg_cost = predicted_costs_k_unscaled[-2]
        avg_k = predicted_costs_k_unscaled[-1]

        # ml_logger.info(f'avg_cost: {avg_cost} avg_k: {avg_k}')

        return round(self.fitness_calculator(avg_cost, avg_k, 0.9, 0.1), 4)

        # total_room_area = 0
        # # 指定数组下标
        # indices = np.arange(1, len(params[0]), self.room_feature_len)
        # # 根据指定下标求和
        # total_room_area = np.sum(params[0][indices])
        # return round(self.fitness_calculator(predicted_costs_k_unscaled[-2]/total_room_area, predicted_costs_k_unscaled[-1], 0.9, 0.1), 4)

    # 选择操作，用于从种群中选择两个个体作为亲代
    @timing_decorator
    def selection(self, population):
        # pdb.set_trace()

        # 从种群中随机选择两个个体
        individuals = random.sample(population, 2)

        # 使用 fitness 函数作为评估标准，选择适应度较高（K值较小）的个体
        return min(individuals, key=self.fitness)

    # 返回参数说明：[要更新位置的值,原位置的值,需要更新的位置,原来的位置]
    @timing_decorator
    def mutate_parameter(self, index, value, feature_type):
        # ml_logger.info(f"feature_type {feature_type} ")
        if feature_type == "window_size":
            mutation_value = random.uniform(-0.01, 0.01)  # 按需调整突变范围
            # new_value = max(0, value + mutation_value)
            new_value = value*(1+mutation_value)
            return new_value, new_value, index, index
        elif feature_type == "wall_material":
            # new_value = random.randint(0, wimWarehouse.get_size() - 1)
            # return 1,0,new_value,index
            return 1, 0, self.get_new_index(feature_type, index, wimWarehouse.get_size()), index
        elif feature_type == "glass_material":
            # new_value = random.randint(0, gmWarehouse.get_size() - 1)
            # return 1,0,new_value,index
            return 1, 0, self.get_new_index(feature_type, index, gmWarehouse.get_size()), index
        elif feature_type == "wf_material":
            # new_value = random.randint(0, wfmWarehouse.get_size() -1)
            # return 1,0,new_value,index
            return 1, 0, self.get_new_index(feature_type, index, wfmWarehouse.get_size()), index
        else:
            raise ValueError("Invalid feature type")

        # return new_value
        return value, value, index, index

    def get_new_index(self, feature_type, index, max_step):

        i = 0
        while i < 10:  # 确保新的index 落在同一个feature 的下标范围内
            step = random.randint(0, max_step - 1)
            n_step = index + step
            if House().find_feature_name(n_step) == feature_type:
                return n_step
            i = i + 1

        # 否则，返回原下标
        return index

    # 交叉操作，用于生成两个后代
    @timing_decorator
    def crossover(self, parent1, parent2):
        # pdb.set_trace()
        # 随机选择一个交叉点，该点位于两个父代个体的有效索引范围内
        # crossover_point = random.randint(0, len(parent1) - 1)
        # crossover_point = random.randint(1, len(parent1) // 12) * 12  # 交叉点必须是12的倍数
        # 确定交叉点
        # 2023.11.24 sunhy，要优化为 houses 、staircases、corridors 对应交叉

        def sub_crossover(p1, p2, base_num):
            # pdb.set_trace()
            # 判断 P1 和 P2 是否是 array
            if not isinstance(p1, np.ndarray):
                p1 = np.array(p1)
            if not isinstance(p2, np.ndarray):
                p2 = np.array(p2)

            crossover_point = random.randint(
                0, base_num-1) * (len(p1) // base_num)
            child1 = np.concatenate(
                (p1[:crossover_point], p2[crossover_point:]))
            # child1 = p1[:crossover_point] + p2[crossover_point:]
            child2 = np.concatenate(
                (p2[:crossover_point], p1[crossover_point:]))
            # child2 = p2[:crossover_point] + p1[crossover_point:]

            return child1, child2

        p1_house_features, p1_stair_features, p1_corridor_features = mdsp.extract_building_features(
            parent1)

        p2_house_features, p2_stair_features, p2_corridor_features = mdsp.extract_building_features(
            parent2)

        child1_house_features, child2_house_features = sub_crossover(
            p1_house_features, p2_house_features, Floor.HOUSE_NUMBER)
        child1_stair_features, child2_stair_features = sub_crossover(
            p1_stair_features, p2_stair_features, Floor.STAIRCASE_NUMBER)
        child1_corridor_features, child2_corridor_features = sub_crossover(
            p1_corridor_features, p2_corridor_features, Floor.CORRIDORS_NUMBER)

        child1 = np.concatenate(
            (child1_house_features, child1_stair_features, child1_corridor_features))
        child2 = np.concatenate(
            (child2_house_features, child2_stair_features, child2_corridor_features))

        # ml_logger.info(f"parent1 len {parent1.shape} parent2 len {parent2.shape} child1 len {child1.shape} child2 len {child2.shape}")
        #ml_logger.info(
        #    f"child1 shape {child1.shape} child2 shape {child2.shape} ")

        return child1, child2

    # 变异操作，用于生成一个新的后代
    @timing_decorator
    def mutation(self, mutated_individual, mutation_rate):

        @timing_decorator
        def mutation_handle(mutated_individual, mutation_rate):

            temp = mutated_individual.copy()
            #ml_logger.info("mutation_handler start", mutated_individual)

            # 10次可能的随机机会,但只能变异一个地方

            for i in range(10):
                if random.uniform(0.0, 0.2) > mutation_rate:
                    continue

                index = random.randint(0, len(mutated_individual)-1)
                feature_type = House().find_feature_name(index)

                #ml_logger.info(
                #    f"mutation_handler mutated_individual INDEX {index} feature_type {feature_type}")

                if feature_type is not None:
                    if feature_type == "No mutation allowed":  # 非可变变异数据区
                        continue
                    # 如果窗口面积为0，跳过突变
                    if feature_type == "window_size" and mutated_individual[index] == 0:
                        continue

                mutated_value, o_value, c_index, o_index = self.mutate_parameter(
                    index, mutated_individual[index], feature_type)
                mutated_individual[o_index] = o_value  # 更新原位置的值
                mutated_individual[c_index] = mutated_value  # 更新新位置的值

                break   # 只变异一个地方

            return mutated_individual

        mutated_individual = np.array([mutated_individual])

        house_features, stair_features, corridor_features = mdsp.extract_building_features(
            mutated_individual)
        # 调用 mutation_handler 循环处理 house_features 中的 12 个 house
        # 创建一个纯 0 的数组，用于存储变异后的 house_features
        new_house_features = np.zeros_like(house_features)

        for i in range(Floor.HOUSE_NUMBER):
            # 截取出一个 house 的特征
            house_f = house_features[i *
                                     House.HOUSE_FEATURES:(i+1)*House.HOUSE_FEATURES]
            mutated_house_features = mutation_handle(house_f, mutation_rate)

            # 将变异后的 house 特征存入 new_house_features
            new_house_features[i*House.HOUSE_FEATURES:(
                i+1)*House.HOUSE_FEATURES] = mutated_house_features

        # 将变异后的 house_features 与 stair_features 和 corridor_features 合并
        mutated_individual = np.concatenate(
            (new_house_features, stair_features, corridor_features)).flatten()

        return [mutated_individual]
    
 
    # 采用mpi 并行计算，每个进程生成一部分种群，提高效率
    # 在脚本中调用时，需要在命令行中指定进程数量，如：mpirun -n 4 python3 GAHousePredictorOnCUDA.py
    # 进程数与 脚本中的 population_size 最好是整数倍，比如 进程为 4，population_size 为 48 

    comm = MPI.COMM_WORLD # 初始化通信器，必须在所有进程中调用
    rank = comm.Get_rank() # 获得当前进程的编号
    rank_size = comm.Get_size() # 获得进程总数
    hostname = MPI.Get_processor_name() # 获取当前节点的名称
    #ml_logger.info(f"MPI hostname: {hostname} rank: {rank} size: {rank_size} ")

    def mpi_info(self,info):
        ml_logger.info(f"MPI hostname[{self.hostname}],Rank[{self.rank}/{self.rank_size}]: {info} ")

    @timing_decorator
    def run(self, json_doc_id=None, outFile=None, url=None):
        #pdb.set_trace()
        """
        运行遗传算法的主函数。
        """
        # 种群的初始化
        begin_time = time.time()
        json_data = couchdb_pool.get_doc(json_doc_id)
        self.mpi_info(f"Get json_data file {json_doc_id} from couchdb")

        # 根据进程数量，计算每个进程需要计算的种群数量,尽可能均匀分布
        sub_population_size = self.population_size // self.rank_size

        # 计算每个进程需要计算的种群数量
        if self.rank < self.population_size % self.rank_size:
            sub_population_size += 1

        json_data.pop("_rev")  # 剔除文档中的 _rev 属性

        self.mpi_info("Generate test population ...")

        # 调用 MagicalDatasetProducer_v2.py 中的 generate_test_population_v2 函数生成种群
        sub_building_x, sub_building_y = mdsp.generate_test_population_v2(f"[{self.rank}/{self.rank_size}]",
             sub_population_size, json_data)
        
        self.mpi_info(
            f"Generate test population :{round(float(time.time()-begin_time),2)} seconds")
        
        if self.rank > 0: # 非root 进程，只将生成的种群序列化，发送到root进程，在root 进程接收完成后，进程自动退出
                    
            data = pickle.dumps(sub_building_x) # 序列化 

            self.mpi_info("Send data to root process ...,data len {}".format(len(data)))
            self.comm.Send(data, dest=0, tag=self.rank) # 发送数据到root 进程
            self.mpi_info("Send data to root process done")
            self.comm.Barrier()    # 等待其他进程发送数据完成，在所有进程未完成发送前，进程会阻塞在此处
            self.mpi_info("Barrier ...done")
            return  # 非root 进程，发送完数据后，退出进程
 
        # 如果是root进程，接收其他进程的数据，生成完整的种群
        if self.rank == 0:
            self.mpi_info("Recv data from other process ...")
            #self.comm.Barrier()
            self.mpi_info("Barrier ...done")
            self.population = []

            self.population.extend(sub_building_x)
            # 从其他进程接收数据
            for source in range(1, self.rank_size):

                data_received_bytes = self.comm.recv(source=source, tag=source)
                #self.mpi_info(f"Recv data from rank {source} done .recv_data len {len(data_received_bytes[0])} ,type {type(data_received_bytes[0])}")
                #self.mpi_info(f"Recv data from rank {source} done")
                self.population.extend(data_received_bytes)
                self.mpi_info(f"Recv data from rank {source} done")
            
            self.comm.Barrier()

            gen_population_time = round(float(time.time()-begin_time),2)
                
        begin_time = time.time()

        exl_filename = mdsp.generate_cn_filename(json_doc_id)
        out_file = outFile if outFile is not None else exl_filename
        ml_logger.info(f"The prediction target excel file name is {out_file}")

        # 将预测对象写入excel表
        j2ec = JSONToExcelConverter(['key'])

        j2ec.json_to_excel(filename=exl_filename, output_filename=out_file,
                           model="DATA",
                           data=json_data,
                           sheet_name="Prediction target", titles=self.excel_file_titles)

        ml_logger.info(
            f"Write prediction target into excel :{round(float(time.time()-begin_time),2)} seconds")
        
        self.population = [[self.population[i].copy()]
                             for i in range(self.population_size)]

        best_individual = None
        begin_time = time.time()
        ml_logger.info(
            f"Running Genetic Algorithm (Inference using TensorRT):{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        # 运行遗传算法
        # for generation in range(self.num_generations):
        for generation in tqdm(range(self.num_generations), desc="Progress"):

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
                child1 = self.mutation(child1, self.mutation_rate)

                # 以突变率的概率对第二个后代进行突变
                '''
                if random.random() < self.mutation_rate:
                    child2 = self.mutation(child2,0.01)
                    '''
                child2 = self.mutation(child2, self.mutation_rate)

                # 将产生的两个后代添加到新种群中
                new_population.extend([child1, child2])

            # 使用新种群替换旧种群
            self.population = new_population

            # 找到具有最佳适应度的个体
            best_individual = min(self.population, key=self.fitness)

            best_fitness = self.fitness(best_individual)
            with self.writer.as_default():
                tf.summary.scalar(
                    'best_fitness', best_fitness, step=generation)
        
        run_ga_time = round(float(time.time()-begin_time),2)

        ml_logger.info(
            f"Run Genetic Algorithm :{run_ga_time} seconds")

        # 关闭 SummaryWriter
        self.writer.close()

        # 使用神经网络进行预测
        # 将最佳个体转换为相应的参数
        best_params = best_individual

        # 对参数进行缩放，以匹配神经网络的输入
        best_params_scaled = self.x_scaler.transform(np.array(best_params).reshape(1, -1))

        # 使用神经网络预测最佳个体的总成本和平均K值
        # predicted_costs_k = self.model.predict([best_params_scaled])[0]
        # 将预测结果逆缩放，以获得实际的总成本和平均K值
        # predicted_costs_k_unscaled = self.y_scaler.inverse_transform([predicted_costs_k])[0]
        # 修改为：推理,on cuda engine 2023.10.19
        rpc_proxy = pool.get_connection()
        predicted_costs_k = rpc_proxy.tensorRT_Service.infer(
            pickle.dumps(best_params_scaled.astype(np.float32)))
        predicted_costs_k = pickle.loads(predicted_costs_k)
        pool.put_connection(rpc_proxy)

        predicted_costs_k_unscaled = self.y_scaler.inverse_transform([predicted_costs_k]).astype(np.float32)[0]

        begin_time = time.time()

        p_building = Building().json_to_building(json_data)

        p_building = p_building.tensor_to_building([np.array(best_params)[0]])

        # 将结果转换为 json，追加写入excel 文件中
        p_building_json = p_building.to_json_cn(json_doc_id)
        j2ec.json_to_excel(filename=exl_filename, output_filename=out_file,
                           data=p_building_json, model="DATA", sheet_name="Prediction result", titles=self.excel_file_titles)

        # 将两个 sheet 中不一样部分自动标识出来
        j2ec.compare_sheets(out_file, "Prediction target", "Prediction result")

        cost_view = p_building.get_cost_view()

        # print(cost_view)
        ml_logger.info(f"Writing pdf {out_file}.pdf ......")

        # 从生成exl 文件 的 “Prediction result” sheet 中提取内容 生成 pdf 文件
        total_pages = excel_to_pdf(excel_file=out_file, sheet_name="Prediction result",
                     pdf_file=f'{out_file}.pdf', summary_data=cost_view, del_title=True)
        
        w_pdf_time = round(float(time.time()-begin_time),2)    
        ml_logger.info(
            f"Writing pdf :{w_pdf_time} seconds")
        
        # 汇总打印整个运行过程的时间和效率，包括生成种群、遗传算法、神经网络推理、生成pdf文件

        ml_logger.info("#################################### Summary ####################################")
        ml_logger.info(f"# Generate test population :{gen_population_time} seconds, {round(float(gen_population_time/self.population_size),2)} s/population,rank_num {self.rank_size}")
        ml_logger.info(f"# Running Genetic Algorithm (Inference using TensorRT):{run_ga_time}, {round(float(run_ga_time/self.num_generations),2)} s/generation")
        ml_logger.info(f"# Writing pdf :{w_pdf_time} seconds, {round(float(total_pages/w_pdf_time),2)} pages/s")
        ml_logger.info("###################################### End ######################################")



        return True
    
@ml_logger.catch(reraise=True)
def main(args):

    ml_logger.level(args.log_level)

    """
    主函数，创建 GAHousePredictor 类的实例并运行遗传算法。
    """
    #start_time = time.time()  # 记录开始时间

    house_predictor = GAHousePredictorOnCUDA(
        args.num_generations, args.population_size, args.mutation_rate)  # , model_inference)

    r = house_predictor.run(args.json_doc_id, args.outFile, args.url)

    #end_time = time.time()  # 记录结束时间
    #elapsed_time = end_time - start_time  # 计算执行时间

    #print(f"Execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run GA House Predictor")

    parser.add_argument('--log_level', type=str, required=True,
                        default="INFO", help='log level')

    parser.add_argument('--json_doc_id', type=str, required=True,
                        default=None, help='json doc id in couchdb')
    parser.add_argument('--outFile', type=str, required=True,
                        default=None, help='Path to the file to write')
    parser.add_argument('--url', type=str, required=True,
                        default=None, help='url to the file to download')
    parser.add_argument('--num_generations', type=int, required=True,
                        default=10, help='number of generation')
    parser.add_argument('--population_size', type=int, required=True,
                        default=20, help='size of population')
    parser.add_argument('--mutation_rate', type=float, required=True,
                        default=0.02, help='rate of mutation')
    args = parser.parse_args()
    main(args)