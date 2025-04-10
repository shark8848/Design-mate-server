from typing import List

from sqlalchemy import Column, Integer, String, Text, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    userId = Column(String(50), primary_key=True)
    #userName = Column(String(50), unique=True, nullable=False)
    userName = Column(String(50), nullable=False)
    organizationCode = Column(String(50), nullable=False)
    password = Column(Text, nullable=False)
    role = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    phone = Column(String(20), unique=True, nullable=False)
    #email = Column(String(100), nullable=False)
    #phone = Column(String(20), nullable=False)
    avator = Column(String(200), nullable=True)
    status = Column(String(1), nullable=False)


class Role(Base):
    __tablename__ = 'roles'
    role = Column(String(50), primary_key=True)
    roleName = Column(String(50), nullable=False)
    status = Column(String(1), nullable=False)

class Permissions(Base):
    __tablename__ = 'permissions'
    id = Column(Integer, primary_key=True, unique=True)
    permissionName = Column(String(100), unique=True, nullable=False)
    description = Column(Text, nullable=False)

class RolePermission(Base):
    __tablename__ = 'role_permissions'
    role = Column(String(50), ForeignKey('roles.role'), primary_key=True)
    permissionName = Column(String(100), ForeignKey('permissions.permissionName'), primary_key=True)

    permission = relationship("Permissions", backref="role_permissions")

class AtomicFunctions(Base):
    __tablename__ = 'atomic_functions'
    functionCode = Column(String(100), primary_key=True)
    functionName = Column(String(100), primary_key=True)
    functionDesc = Column(Text)
    url = Column(String(200))
    api = Column(String(200))
    status = Column(String(1), nullable=False)

class Functions(Base):
    __tablename__ = 'functions'
    functionCode = Column(String(100), primary_key=True)
    functionName = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    url = Column(String(200))
    api = Column(String(200))
    subFunctions = Column(Text)
    status = Column(String(1), nullable=False)

class PermissionFunctions(Base):
    __tablename__ = 'permission_functions'
    permissionName = Column(String(100), ForeignKey('permissions.permissionName'), primary_key=True)
    functionCode = Column(String(100), primary_key=True)
    subfunctions = Column(Text)

class Menus(Base):
    __tablename__ = 'menus'

    menuId = Column(String(100), primary_key=True)
    menuDesc = Column(String(100), nullable=False)
    flag = Column(String(1), nullable=False)
    fatherMenuId = Column(Integer)
    functions = Column(Text)
    icon = Column(String(200))
    status = Column(String(1), nullable=False)
