from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# 定义 HouseTemplate 对象
class HouseTemplate(Base):
    __tablename__ = 'house_template'

    templateId = Column(String(100), primary_key=True)
    organizationCode = Column(String(20), nullable=False)
    projectId = Column(String(20), nullable=False)
    buildingId = Column(String(20))
    templateName = Column(String(20), nullable=False)
    templateJson = Column(String(100), unique=True, nullable=False)
    status = Column(Integer, nullable=False)

# 定义 HouseInstance 对象
class HouseInstance(Base):
    __tablename__ = 'house_instance'

    instanceId = Column(String(100), primary_key=True)
    organizationCode = Column(String(20), nullable=False)
    projectId = Column(String(20), nullable=False)
    buildingId = Column(String(20), nullable=False)
    templateId = Column(String(100), nullable=False)
    instanceName = Column(String(20), nullable=False)
    instanceJson = Column(String(100), unique=True, nullable=False)
    status = Column(Integer, nullable=False)
