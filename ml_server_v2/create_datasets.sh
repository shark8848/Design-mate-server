# Author sunhy 2023.10.7
# options:
#  -h, --help            show this help message and exit
#  --spark_master_url SPARK_MASTER_URL
#                        Spark Master URL
#  --log_level LOG_LEVEL
#                        log level: INFO, DEBUG, WARN, ERROR
#  --num_partitions NUM_PARTITIONS
#                        number of partitions
#  --data_size DATA_SIZE
#                        number of data to generate
#  --datasets_dir DATASETS_DIR
#                        dataset directory
#  --py_lib PY_LIB       python lib path,call ../push_to_hdfs.sh to gen python_lib.tar and push to hdfs
#  --hdfs_url HDFS_URL   url of hdfs
#  --hdfs_user HDFS_USER
#                        user of hdfs
#  --hdfs_dataset_dir HDFS_DATASET_DIR
#                        hdfs dataset directory

# 调用上一级目录下的脚本生成 python_lib.tar
##	--append_data
cd ..
sh push_to_hdfs.sh "*.pkl"
cd ml_server_v2
python DataSetGenerator.py \
	--spark_master_url 	spark://192.168.1.19:7077 \
	--log_level		INFO \
	--num_partitions	8 \
	--data_size		256 \
	--datasets_dir		./datasets/ \
	--py_lib		hdfs://192.168.1.19:9000/dependency/python_lib.tar \
	--hdfs_url		http://192.168.1.19:9870 \
	--hdfs_user		root \
	--hdfs_datasets_dir	/datasets \
	--append_data
