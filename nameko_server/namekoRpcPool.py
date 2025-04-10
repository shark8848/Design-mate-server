sys.path.append("..")
from apocolib.apocolog4p import apoLogger as apolog
from apocolib import RpcProxyPool

class namekoRpcPool(RpcProxyPool):

     @staticmethod
     def getConnection:
         return self.get_connection()
     def releaseConnection(conn):
         return self.put_connection(conn)
