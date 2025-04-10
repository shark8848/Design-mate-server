from nameko.rpc import rpc
from sqlalchemy import create_engine, Column, String, Integer, CHAR
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from urpfModel import *
import pdb
import json
import sys
sys.path.append("..")
#from apocolib.apocolog4p import apoLogger as apolog
from apocolib.NamekoLogger import namekoLogger as nameko_logger
from apocolib import sqliteSession as sqlSession

class permissionsService:

    name = "permissionsService"

    # 获取用户所拥有的角色和权限
    def get_user_roles_permissions(self,user_id):
#        pdb.set_trace()

        roles = None
        permissions = None
        with sqlSession.sqliteSession().getSession() as session:

            userRoles,organizationCode = session.query(
                    User.role,
                    User.organizationCode
            ).filter(User.userId == user_id).first()

            roles = userRoles.split(',')
            # 查询角色的权限和功能
            permissions = []

            for role in roles:
                role_permissions = session.query(
                        RolePermission.permissionName
                ).filter(RolePermission.role == role).all()

                for permission in role_permissions:
                    if permission.permissionName not in permissions:
                        permissions.append(permission.permissionName)

        return roles, organizationCode, permissions

    # 查询functioncode 所在菜单
    def get_menu(self,function_code):

        #pdb.set_trace()

        menu = None
        with sqlSession.sqliteSession().getSession() as session:
            menu = session.query(
                    Menus.menuDesc
            ).filter(Menus.functions.like(f'%{function_code}%')).first()

        return menu[0]

    @rpc
    def get_user_roles_permissions_functions(self, user_id) :#-> dict:
        try:
            # pdb
            roles, organizationCode, role_permissions = self.get_user_roles_permissions(user_id)
            menus = []
#            pdb.set_trace()

            for i,permission in enumerate(role_permissions):

                # 根据permission_name 查询 functions && subfunctions
                with sqlSession.sqliteSession().getSession() as session:
                    functions = session.query(
                        PermissionFunctions.permissionName,
                        PermissionFunctions.functionCode,
                        PermissionFunctions.subfunctions,
                        Functions.description,
                        Functions.url,
                        Functions.api
                    ).join(PermissionFunctions, PermissionFunctions.functionCode == Functions.functionCode).filter(
                        PermissionFunctions.permissionName == permission
                    ).all()

                    tmp_func = None
                    menu_data = None
                    temp_menu = None

                    for j,function in enumerate(functions):

                        menu_desc = self.get_menu(function.functionCode)
                        if temp_menu is None or menu_desc != temp_menu:
                        #if tmp_func is None or function.functionCode != tmp_func:
                            menu_data = {
                                "menu": menu_desc,
                                "items": []
                            }
                            temp_menu = menu_desc
                        t_functions = {"functionName":function.description,"url":function.url,"subfunctions":[]}


                        if function.subfunctions is None or function.subfunctions =='':
                            continue
                        subFunction = function.subfunctions.split(',')
                        for subFunc in subFunction:
                            atomicFunctions = session.query(
                                    AtomicFunctions.functionName,
                                    AtomicFunctions.functionDesc,
                                    AtomicFunctions.url,
                                    AtomicFunctions.api
                            ).filter(AtomicFunctions.functionCode == subFunc).all()[0]
                            t_functions["subfunctions"].append(
                                    {
                                        "api":atomicFunctions.api,
                                        "functionDesc":atomicFunctions.functionName,
                                        "url":atomicFunctions.url,
                                        "desc":atomicFunctions.functionDesc
                                    }
                            )
                        #-end for sunfunc
                        #menu_data["items"].append(permissions[i])
                        menu_data["items"].append(t_functions)
                    # end for function
                    if menu_data is not None:
                        menus.append(menu_data)
                #end with
            #for end - role-permissions
            result = {"userId": user_id, "organizationCode": organizationCode, "roles": roles, "permissions": menus }

            return 0,result,f"get {user_id}'s permission successfully"

        except Exception as e:
            nameko_logger.error(f'get {user_id} permission error.{str(e)}')
            return -1, [],f'get {user_id} permission failed: {str(e)}'
