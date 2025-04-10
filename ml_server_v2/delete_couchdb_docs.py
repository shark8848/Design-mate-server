import requests
import json
import sys

# 确保传入了正确的参数
if len(sys.argv) != 2:
    print("Usage: python script.py <search_string>")
    sys.exit(1)

search_string = sys.argv[1]

# 获取匹配的文档列表，包括 _id 和 _rev
response = requests.get("http://apoco:apoco2023@couchdb.apoco.com.cn/design_mate/_all_docs")
data = response.json()

# 从清单文档中读取文档 ID 和 _rev，并逐个删除
with open("document_list.txt", "w") as f:
    for row in data.get("rows", []):
        if search_string in row["key"]:
            doc_id = row["id"]
            doc_rev = row["value"]["rev"]
            f.write(f"{doc_id} {doc_rev}\n")
            print(f"Deleting document with ID: {doc_id} and Rev: {doc_rev}")
            requests.delete(f"http://apoco:apoco2023@couchdb.apoco.com.cn/design_mate/{doc_id}?rev={doc_rev}")
