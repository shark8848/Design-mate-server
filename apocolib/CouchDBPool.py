import couchdb
import pdb
import requests

class CouchDBPool:
    def __init__(self, server_url, db_name):
        # 连接到 CouchDB 服务器
        self.server = couchdb.Server(server_url)
        self.connection_pool = {}
        self.db_name = db_name

    def get_db(self):
        # 获取数据库连接，如果不存在则创建新的连接
        if self.db_name in self.connection_pool:
            return self.connection_pool[self.db_name]
        else:
            if self.db_name in self.server:
                db = self.server[self.db_name]
            else:
                db = self.server.create(self.db_name)
            self.connection_pool[self.db_name] = db
            return db

    def insert_doc(self, doc):
        # 插入文档
        db = self.get_db()
        #print("------------------------ insert_doc ---------------------------")
        #print(doc)
        return db.save(doc)

    def update_doc(self, doc_id, new_data):
        # 更新文档
        db = self.get_db()
        doc = db[doc_id]
        doc.update(new_data)
        return db.save(doc)

    def delete_doc(self, doc_id):
        # 删除文档
        db = self.get_db()
        doc = db[doc_id]
        return db.delete(doc)

    def get_doc(self, doc_id):
        # 获取文档
        db = self.get_db()
        return db.get(doc_id)

    def get_doc_json(self, doc_id):

        response = requests.get(f"{couch_dbpool_url}/{db_name}/{doc_id}")
        # 检查响应状态码
        if response.status_code == 200:
            # 提取文档的JSON数据
            doc_json = response.json()
            print(doc_json)
            return doc_json
        else:
            print(f"Failed to retrieve document. Status code: {response.status_code}")

        return None

    def query_view(self, view_name):
        # 使用视图查询文档
        db = self.get_db()
        return db.view(view_name)

    '''
    def query_view(self, design_doc, view_name, **kwargs):
        # 使用视图查询文档
        db = self.get_db()
        return db.view(design_doc, view_name, **kwargs)
        '''

couch_dbpool_url = 'http://admin:apoco2024@couchdb.apoco.com.cn'
db_name = 'design_mate'
couchdb_pool = CouchDBPool(couch_dbpool_url,db_name)

def main():
    pdb.set_trace()
    couch_dbpool_url = 'http://admin:apoco2024@couchdb.apoco.com.cn'
    db_name = 'design_mate'
    couchdb_pool = CouchDBPool(couch_dbpool_url, db_name)

    # 插入文档
    doc = {'_id': '00001','name': 'John Doe', 'age': 30}
    doc_id, _ = couchdb_pool.insert_doc(doc)
    print(f"Inserted document with ID: {doc_id}")

    # 查询文档
    #doc_id = 'your_document_id'  # 请将其替换为实际存在的文档 ID
    doc = couchdb_pool.get_doc_json(doc_id)
    if doc:
        print(f"Document with ID {doc_id}: {doc}")
    else:
        print(f"Document with ID {doc_id} not found.")
    print(f"doc id {doc_id} {doc} ")

    # 更新文档
    #doc_id = 'your_document_id'  # 请将其替换为实际存在的文档 ID
    new_data = {'age': 31}
    couchdb_pool.update_doc(doc_id, new_data)
    print(f"Document with ID {doc_id} updated successfully.")

    # 删除文档
    #doc_id = 'your_document_id'  # 请将其替换为实际存在的文档 ID
    couchdb_pool.delete_doc(doc_id)
    print(f"Document with ID {doc_id} deleted successfully.")
    '''
    # 查询视图
    design_doc = 'your_design_doc'  # 请将其替换为实际存在的设计文档
    view_name = 'your_view_name'    # 请将其替换为实际存在的视图名
    results = couchdb_pool.query_view(design_doc, view_name)
    for row in results:
        print(f"Document with key '{row.key}': {row.value}")
        '''

    try:
        # 使用视图查询
        view_name = '_design/_room_view/_view/_room_view'
        view_result = couchdb_pool.query_view(view_name)

        # 遍历查询结果并输出 rooms 的内容
        for row in view_result:
            room = row.value
            print("Room Name:", room["room_name"])
            print("Room Type:", room["room_type"])
            print("Room Area:", room["room_area"])
            # ... （根据实际数据结构继续打印其它房间属性）
            print("-" * 30)

    except Exception as e:
        print("Error querying rooms:", e)

if __name__ == "__main__":
    main()
