import sqlite3, os, time
import traceback
import apocolib.sqlite_config as config


class Sqlite():
    db_file = None  # 数据库文件
    connection = None  # 数据库连接对象

    def __init__(self):
        self.db_file = config.SQLITE_DBFILE

    # 获取数据库对象
    def GetConn(self):
        try:
            if self.connection == None:
                self.connection = sqlite3.connect(self.db_file)
                self.connection.text_factory = str
        except Exception as ex:
            traceback.print_exc()
            return "error: " + str(ex)

    # 增加任务
    def insert(self, sql):
        self.write_lock()
        self.GetConn()
        self.connection.text_factory = str
        try:
            result = self.connection.execute(sql)
            id = result.lastrowid
            self.connection.commit()
            self.rm_lock()
            return id
        except Exception as ex:
            return "error: " + str(ex)

    # 删除任务
    def delete(self, sql):
        self.write_lock()
        self.GetConn()
        try:
            result = self.connection.execute(sql)
            self.connection.commit()
            self.rm_lock()
            return result.rowcount
        except Exception as ex:
            return "error: " + str(ex)

    # 修改任务状态（完成/未完成）
    def update(self, sql):
        self.GetConn()
        try:
            result = self.connection.execute(sql)
            self.connection.commit()
            return result.rowcount
        except Exception as ex:
            return "error: " + str(ex)

    # 查询任务
    def select(self, sql):
        self.GetConn()
        result = self.connection.execute(sql)
        data = result.fetchall()
        tmp = list(map(list, data))  # 元组转成列表
        data = tmp
        return data

    # 方式1：创建数据表
    def create(self):
        self.GetConn()
        sql = '''create table if not exists tb_user(
    	          ID INT PRIMARY KEY     NOT NULL,
                  NAME           TEXT    NOT NULL,
                  AGE            INT     NOT NULL,
                  ADDRESS        CHAR(50),
                  SALARY         REAL);'''
        try:
            self.connection.execute(sql)
            self.connection.commit()
            self.connection.close()
        except Exception as ex:
            traceback.print_exc()
            return "error: " + str(ex)

    # 方式2：创建数据表
    def create_back(self):
        self.GetConn()
        try:
            fr = open('/www/wwwroot/ding-server/db/task.sql', 'rb')
            result = self.connection.executescript(fr.read())
            fr.close()
            self.connection.commit()
            return True
        except Exception as ex:
            traceback.print_exc()
            return False

    def writeFile(self, filename, s_body, mode='w+'):
        try:
            fp = open(filename, mode);
            fp.write(s_body)
            fp.close()
            return True
        except:
            try:
                fp = open(filename, mode, encoding="utf-8");
                fp.write(s_body)
                fp.close()
                return True
            except:
                return False

    # 是否有锁
    def is_lock(self):
        n = 0
        while os.path.exists(self.__LOCK):
            n += 1
            if n > 50:
                self.rm_lock()
                break
            time.sleep(0.01)

    # 写锁
    def write_lock(self):
        self.is_lock()
        self.writeFile(self.__LOCK, "True")

    # 解锁
    def rm_lock(self):
        if os.path.exists(self.__LOCK):
            os.remove(self.__LOCK)


if __name__ == '__main__':
    obj = Sqlite()
    # obj.GetConn()    # 获取数据库连接，没有的话，就创建数据库
    obj.create()
