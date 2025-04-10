from nameko.rpc import rpc
from sqlalchemy import create_engine, Column, String, Integer, CHAR
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import sys
sys.path.append("..")
#from apocolib.apocolog4p import apoLogger as apolog
from apocolib.NamekoLogger import namekoLogger as nameko_logger
from apocolib import sqliteSession as sqlSession
from urpfModel import Role

# 定义 "roles" 表的映射类
'''
Base = declarative_base()

class Role(Base):
    __tablename__ = "roles"
    role = Column(String(50), primary_key=True)
    roleName = Column(String(50), nullable=False)
    status = Column(CHAR(1), nullable=False)
'''
# 定义提供 Nameko 微服务的类
class rolesService:
    name = "rolesService"

    # 查询所有角色
    @rpc
    def get_all_roles(self):

        try:
            with sqlSession.sqliteSession().getSession() as session:

                roles = session.query(Role).all()
                return 0,[{
                    "role": role.role,
                    "roleName": role.roleName,
                    "status": role.status
                } for role in roles],'get roles successfully'

        except Exception as e:
            nameko_logger.error(f'get roles error.{str(e)}')
            return -1, [],f'get roles failed: {str(e)}'
