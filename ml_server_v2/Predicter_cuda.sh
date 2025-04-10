# 
# Author sunhy,2023.10.8
#
#description="Run GA House Predictor")
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

  python AC2NNetPredicterOnCUDA.py \
    --log_level "DEBUG" \
    --json_doc_id "$json_doc_id" \
    --outFile "./predicted_data/${json_doc_id}_${random_sequence}.xlsx" \
    --url "None" \
    --num_generations 10 \
    --population_size 64 \
    --mutation_rate 0.02
else
  echo "Error: Model-json_doc_id $json_doc_id is not found on server $couchdb_url"
fi
