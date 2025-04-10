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

class permissionsService_new:

#    name = "permissionsService_new"


# 获取用户所拥有的角色和权限
    def get_user_roles_permissions(self,user_id):
        pdb.set_trace()

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

        return roles, permissions

    # 生成菜单和权限的JSON
    #@rpc
    def generate_menu_permission_json(self,user_id):
        pdb.set_trace()
        # 获取用户所拥有的角色和权限
        roles, permissions = self.get_user_roles_permissions(user_id)

        # 根据权限生成菜单和功能列表
        menus = []
        for permission in permissions:
            permission_functions = PermissionFunction.objects.filter(permissionName=permission)
            for permission_function in permission_functions:
                functions = permission_function.subfunctions.split(',')
                for function in functions:
                    atomic_functions = AtomicFunction.objects.get(functionCode=function)
                    menu = Menu.objects.get(menuId=atomic_functions.menuId)
                    menu_data = {
                        "menu": menu.menuDesc,
                        "items": []
                    }
                    for item in menu.functions.split(','):
                        if item in functions:
                            function_data = {}
                            atomic_function = AtomicFunction.objects.get(functionCode=item)
                            function_data["functionName"] = atomic_function.functionName
                            function_data["subfunctions"] = []
                            subfunctions = atomic_function.subFunctions.split(',')
                            for subfunction in subfunctions:
                                subfunction_data = {}
                                atomic_subfunction = AtomicFunction.objects.get(functionCode=subfunction)
                                subfunction_data["api"] = atomic_subfunction.api
                                subfunction_data["desc"] = atomic_subfunction.functionDesc
                                subfunction_data["functionDesc"] = atomic_subfunction.functionDesc
                                subfunction_data["url"] = atomic_subfunction.url
                                function_data["subfunctions"].append(subfunction_data)
                            function_data["url"] = atomic_function.url
                            menu_data["items"].append(function_data)
                    menus.append(menu_data)

        # 将菜单和权限生成JSON
        result = {"permissions": []}
        for menu in menus:
            added_menu = False
            for permission in result["permissions"]:
                if menu["menu"] == permission["menu"]:
                    added_menu = True
                    for item in menu["items"]:
                        if item["resources"] == permission["items"][0]["resources"]:
                            permission["items"][0]["functions"].append(item["functions"][0])
            if not added_menu:
                menu_data = {
                    "menu": menu["menu"],
                    "items": [
                        {
                            "resources": menu["items"][0]["resources"],
                            "functions": menu["items"]
                        }
                    ]
                }
                result["permissions"].append(menu_data)

        #return json.dumps(result, ensure_ascii=False, indent=4)
        return result

if __name__ == '__main__':
    pdb.set_trace()
    ps = permissionsService_new()
    result = ps.generate_menu_permission_json('sunzimo')


