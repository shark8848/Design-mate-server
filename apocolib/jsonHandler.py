import json
import pdb

class jsonHandler:

    def __init__(self, file_path):
        print(f'{file_path}')
        self.file_path = file_path
        self.data = self.load_data()
        print(f'{self.data}')

    def load_data(self):
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError as e:
            print(f'str(e)')
            return {}

    def save_data(self):
        try:
            with open(self.file_path, 'w') as f:
                # 获取文件锁
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
               #json.dump(data,f,indent=4)
                json.dump(self.data,f,indent=4,ensure_ascii=False)
                # 释放文件锁
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return True
        except:
            return False

    def search(self, keys, values):
        #pdb.set_trace()
        results = []
        for item in self.data:
            if not isinstance(item, dict):
                continue
            match = True
            for key, value in zip(keys, values):
                if item.get(key) != value:
                    match = False
                    break
            if match:
                results.append(item)
        return results

    def insert(self, new_item):
        self.data.append(new_item)
        return self.save_data()

    def delete(self, keys, values):
        new_data = []
        for item in self.data:
            match = True
            for key, value in zip(keys, values):
                if item.get(key) != value:
                    match = False
                    break
            if not match:
                new_data.append(item)
        self.data = new_data
        return self.save_data()

    def update(self, keys, values, new_item):
        for item in self.data:
            match = True
            for key, value in zip(keys, values):
                if item.get(key) != value:
                    match = False
                    break
            if match:
                item.update(new_item)
                return self.save_data()
        return False
