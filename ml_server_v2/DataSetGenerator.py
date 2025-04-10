import sys
sys.path.append("..")
from apocolib.dataset_io import save_dataset, load_dataset, append_to_dataset
from apocolib.GPUConfigurator import GPUConfigurator
from apocolib.MlLogger import mlLogger as ml_logger
from ml_server_v2.BuildingElement import (generate_space_numbers,
                                          Floor, Building, BuildingCreator, save_to_couchdb, publicSpaceCreator)
from ml_server_v2.BuildingSpaceBase import HousesCreator
import argparse
import time
import tensorflow as tf
import numpy as np
import pandas as pd
import json
from datetime import datetime
import pdb
import copy
import random
import os
import random
import hdfs

import pickle
from pyspark.sql import SparkSession

# gpu_memory_limit = int(os.getenv('GPU_MEMORY_LIMIT', '1024'))
# # 配置gpu 及内存使用大小
# configurator = GPUConfigurator(use_gpu=True, gpu_memory_limit=gpu_memory_limit)
# configurator.configure_gpu()
# tf.device(configurator.select_device())


class DataSetGenerator:

    # 每层最大空间数
    MAX_SPACE_PER_FLOOR = 12
    # 每层最小空间数
    MIN_SPACE_PER_FLOOR = 6

    MAX_FLOOR_NUM = 34
    MIN_FLOOR_NUM = 12

    def __init__(self, data_size=1000):
        self.logger = ml_logger
        self.logger.info("DataSetGenerator init")
        self.data_size = data_size

        self.data_set_x = []
        self.data_set_y = []

    def generate_dataset(self):

        self.logger.info("DataSetGenerator generate_dataset")

        for i in range(self.data_size):

            house_num = random.randint(
                self.MIN_SPACE_PER_FLOOR, self.MAX_SPACE_PER_FLOOR)  # 随机每层房间数
            hc = HousesCreator(house_num)
            houses = hc.make_houses()

            # 创建building,并保存到couchdb,从每个building中随机取第4层数据作为训练数据

            floor_num = random.randint(
                self.MIN_FLOOR_NUM, self.MAX_FLOOR_NUM)  # 随机楼层数
            bc = BuildingCreator(floor_num, house_num)
            building = bc.create_building(
                buildingName=f"building {i}", houses=houses)

            doc_id = save_to_couchdb(building)
            if doc_id:
                print(f"Doc_ID-[{doc_id}] Save to couchdb successfully ")
            else:
                print("Save to couchdb failed")

            features, target = building.get_floor_tensor(4)  # 取 第 4 层数据

            self.data_set_x.append(features)
            self.data_set_y.append(target)

        return self.data_set_x, self.data_set_y


def generate_building(_):

    house_num, staircase_num, corridor_num = generate_space_numbers()

    # house_num = random.randint(6, 12)
    hc = HousesCreator(house_num)
    houses = hc.make_houses()

    floor_num = random.randint(5, 10)
    #bc = BuildingCreator(floor_num, house_num)
# --------
    sc = publicSpaceCreator(staircase_num, space_code=12)  # 生成楼梯
    staircases = sc.make_public_spaces()

    cc = publicSpaceCreator(corridor_num, space_code=13)  # 生成走廊
    corridors = cc.make_public_spaces()

    bc = BuildingCreator(floor_num, house_num)
    building = bc.create_building(buildingName=f"building {_}", houses=houses,
                                  staircases=staircases, corridors=corridors)
# -

    # building = bc.create_building(
    #    buildingName=f"building {_}", houses=houses)

    doc_id = save_to_couchdb(building)
    print(f"save_to_couchdb {doc_id}")

    return building, doc_id


def process_building(building_doc_id):
    building, doc_id = building_doc_id
    features, target = building.get_floor_tensor(4)
    return (features, target)


def upload_to_hadoop(hdfs_url='http://192.168.1.19:9870', hdfs_user='root', hdfs_path=None, local_path=None):
    client = hdfs.InsecureClient(hdfs_url, hdfs_user)
    client.upload(hdfs_path, local_path, cleanup=True, overwrite=True)


py_lib = "hdfs://192.168.1.19:9000/dependency/python_lib.tar"


def main(args):

    gpu_memory_limit = int(os.getenv('GPU_MEMORY_LIMIT', '1024'))
    # 配置gpu 及内存使用大小
    configurator = GPUConfigurator(use_gpu=True, gpu_memory_limit=gpu_memory_limit)
    configurator.configure_gpu()
    tf.device(configurator.select_device())

    spark = SparkSession.builder.master(args.spark_master_url).appName("DataSetGenerator").config(
        "spark.serializer", "org.apache.spark.serializer.KryoSerializer").getOrCreate()
    spark.sparkContext.setLogLevel(args.log_level)
    spark.sparkContext.addPyFile(args.py_lib)

    logger = spark._jvm.org.apache.log4j.LogManager.getLogger(
        "DataSetGenerator")
    start_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    logger.info(f" Spark session created @ {start_time}")

    data_size = args.data_size
    data_rdd = spark.sparkContext.parallelize(
        range(data_size), args.num_partitions)
    buildings_and_ids = data_rdd.map(generate_building).collect()

    logger.info("##############################################################")
    logger.info("data_rdd.map(generate_building).collect() completed")
    logger.info("##############################################################")

    data = buildings_and_ids  # Your list of buildings and doc_ids
    logger.info(f"buildings_and_ids size is :{sys.getsizeof(data)}")
    rdd = spark.sparkContext.parallelize(data, args.num_partitions)
    result_rdd = rdd.map(process_building)
    result_data = result_rdd.collect()

    end_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    logger.info("##############################################################")
    logger.info(f"spark session created @ {start_time} , stopped @ {end_time}")
    logger.info("##############################################################")
    spark.stop()

    data_set_x = []
    data_set_y = []

    for item in result_data:
        features, target = item

        data_set_x.append(tf.cast(features, dtype=tf.float64))
        data_set_y.append(tf.cast(target, dtype=tf.float64))

    logger.info(
        f"created trainning tensor features & target @ {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")
    logger.info(
        f"features shape = {np.array(data_set_x).shape} ,target shape = {np.array(data_set_y).shape}")

    if args.append_data:
        append_to_dataset(f'{args.datasets_dir}/data_set_x.pkl', data_set_x)
        append_to_dataset(f'{args.datasets_dir}/data_set_y.pkl', data_set_y)
    else:
        save_dataset(f'{args.datasets_dir}/data_set_x.pkl', data_set_x)
        save_dataset(f'{args.datasets_dir}/data_set_y.pkl', data_set_y)

    upload_to_hadoop(hdfs_url=args.hdfs_url, hdfs_user=args.hdfs_user,
                     hdfs_path=f'{args.hdfs_datasets_dir}/data_set_x.pkl', local_path=f'{args.datasets_dir}/data_set_x.pkl')
    upload_to_hadoop(hdfs_url=args.hdfs_url, hdfs_user=args.hdfs_user,
                     hdfs_path=f'{args.hdfs_datasets_dir}/data_set_y.pkl', local_path=f'{args.datasets_dir}/data_set_y.pkl')


if __name__ == "__main__":

    # 从本地读出 hostname 为 spark_master 的ip地址
    spark_master_ip = '192.168.1.19'
    with open("/etc/hosts", "r") as f:
        for line in f.readlines():
            if "spark_master" in line:
                spark_master_ip = line.split()[0]
                print("get spark master ip ", spark_master_ip)
                break

    parser = argparse.ArgumentParser(
        description="Dataset Generation and Storage")
    parser.add_argument("--spark_master_url", type=str, required=True,
                        default=f"spark://{spark_master_ip}:7077", help="Spark Master URL")
    parser.add_argument("--log_level", type=str, default="INFO", required=True,
                        help="log level: INFO, DEBUG, WARN, ERROR")
    parser.add_argument("--num_partitions", type=int, required=True,
                        default=8, help="number of partitions")
    parser.add_argument("--data_size", type=int, default=32, required=True,
                        help="number of data to generate")
    parser.add_argument("--datasets_dir", type=str, required=True,
                        default="./datasets", help="dataset directory")
    parser.add_argument("--py_lib", type=str, required=True,
                        default=py_lib, help="python lib path")
    parser.add_argument("--hdfs_url", type=str, required=True,
                        default="http://192.168.1.19:9870", help="url of hdfs")
    parser.add_argument("--hdfs_user", type=str, required=True,
                        default="root", help="user of hdfs")
    parser.add_argument("--hdfs_datasets_dir", type=str, required=True,
                        default="/datasets", help="hdfs dataset directory")
    parser.add_argument("--append_data", action="store_true",
                        help="Append data to the existing dataset (if specified)")
    args = parser.parse_args()

    print("args ", args)

    main(args)

