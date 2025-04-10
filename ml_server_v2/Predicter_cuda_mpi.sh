# 
# Author sunhy,2023.10.8
#
# description="Run GA House Predictor"
# 用于多机多进程处理生成种群，并进行整层的参数优化
#--spark_master_url, type=str, required=True,
#                        default=f"spark://{spark_master_ip}:7077", help="Spark Master URL")
#--loadFromFile, type=str, required=True,
#            default=None, help='Path to the file to load from')
#--outFile, type=str, required=True,
#            default=None, help='Path to the file to write')
#--url, type=str, required=True,
#            default=None, help='url to the file to download')
#--num_generations, type=int, required=True,
#            default=10, help='number of generation')
#--population_size, type=int, required=True,
#            default=20, help='size of population')
#--mutation_rate, type=float, required=True,
#            default=0.02, help='rate of mutation')

#!/bin/bash
cd ..
sh push_to_hdfs.sh *.pkl
cd ml_server_v2
# 默认的 json_doc_id 值
default_json_doc_id="16988217596456674"

if [ -z "$1" ]; then
  json_doc_id="$default_json_doc_id"
  echo "INFO: No 'model-json_doc_id' is provided, the system will use a default test model-json_doc_idi:$json_doc_id"
else
  json_doc_id=$1
fi

echo "Predicter_cuda will load model-json_doc_id :$json_doc_id"

couchdb_url="http://admin:apoco2024@couchdb.apoco.com.cn"  
database_name="design_mate"  

# 发送 HEAD 请求来检查文档是否存在
head_response=$(curl -I -s -o /dev/null -w "%{http_code}" "$couchdb_url/$database_name/$json_doc_id")

# 检查 HTTP 响应码
if [ "$head_response" -eq 200 ]; then
  # 文档存在，生成随机序列
  generate_sequence() {
    current_datetime=$(date "+%Y%m%d%H%M%S")
    random_number=$((RANDOM % 10000))
    formatted_random_number=$(printf "%04d" $random_number)
    sequence="${current_datetime}_${formatted_random_number}"
    echo $sequence
  }

  # 调用函数生成序列
  random_sequence=$(generate_sequence)



  # 配置环境变量，设置通信协议
  export OMPI_MCA_btl=self,sm,tcp
  # 配置环境变量，设置发送和接收缓冲区的大小
  export OMPI_MCA_btl_tcp_eager_limit=52428800
  # 参数说明,--mca btl tcp,self 表示使用 tcp 协议进行通信
  # mpiexec -n 12 --mca btl tcp,self -host hadoop2:6,hadoop4:6 \
  # 单机多进程处理,n 不能大于系统最大 slot 数量，使用lscpu 可查
  # 特别要注意指定网卡，在多网卡或者虚拟网卡的情况下，不指定网卡会出现通信错误 --mca btl_tcp_if_include eno1
  # HOROVOD 官网 命令参数参考 https://horovod.readthedocs.io/en/stable/mpi.html
  # MPI 官网 https://www.open-mpi.org/faq/?category=tcp

  mpiexec -np 8 \
    --allow-run-as-root \
    -H hadoop2:4,hadoop4:4 \
    -bind-to none -map-by slot \
    -mca plm_rsh_args "-p 22" \
    --mca btl_tcp_if_include eno1 \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    --mca pml ob1 --mca btl tcp,self \
    /home/apoco/apoco-intelligent-analysis/venv/bin/python AC2NNetPredicterOnCUDA.py \
    --log_level "DEBUG" \
    --json_doc_id "$json_doc_id" \
    --outFile "./predicted_data/${json_doc_id}_${random_sequence}.xlsx" \
    --url "None" \
    --num_generations 20 \
    --population_size 16 \
    --mutation_rate 0.02
else
  echo "Error: Model-json_doc_id $json_doc_id is not found on server $couchdb_url"
fi
