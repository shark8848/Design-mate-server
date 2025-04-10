# -*- coding: utf-8 -*-
"""
This is the main program of Apoco Intelligent Analytics Flask API Server v1.0 ,that requires Python 3.7 or above.
Author: sunhy
Version: 1.0
Date: 2023-01-20
"""
# Check Python version
import sys
if sys.version_info < (3, 7):
    sys.exit("Python 3.7 or above is required to run this program.")

from nameko.standalone.rpc import ClusterRpcProxy
#from datetime import datetime
from flask import Flask,g,Blueprint,render_template,request,redirect,jsonify,make_response,send_file
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, create_refresh_token, get_jwt_identity, get_jwt, decode_token
from gevent import pywsgi
from flask_bcrypt import Bcrypt as bcrypt
import redis
import json,os,signal
import pickle
import base64
import jwt
import datetime
from functools import wraps
from multiprocessing import Process

sys.path.append("..")
#from apocolib.apocolog4p import apoLogger as apolog
#from apocolib.apocolog4p import auditLogger as adtlog
from apocolib.FlaskLogger import flask_log
from apocolib import apocoIAServerConfigurationManager as iaConMg
from apocolib import RpcProxyPool
from apocolib.RedisPool import redisConnectionPool as rcPool
from apocolib.arrayCopy import arrayCopy,arrayCopyByRangeException

# 初始化Flask应用程序
app = Flask(__name__)

SECRET_KEY = iaConMg.flask_jwt_secret_key
JWT_EXPIRATION_DELTA = iaConMg.flask_jwt_expiration_delta
# rpc pool
pool = RpcProxyPool.RpcProxyPool()

# 用户登录验证装饰器
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        clientip = request.headers.get('X-Forwarded-For', request.remote_addr)
        #apolog.info(f'token_required ,client ip {clientip}')
        redis_conn = None
        try:
            if 'Authorization' in request.headers:
                #apolog.info(f"Authorization {request.headers['Authorization']}")
                token = request.headers['Authorization'].split(' ')[1]
                #token = request.headers['Authorization']

            if not token:
                return jsonify({'errorCode': -1,'msg':'Token is missing!'})

            #apolog.info(f' receive token {token} ')

            data = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            #apolog.info(f'token_required {data}')
            current_user = data['username']
            expiry_date = datetime.datetime.fromtimestamp(data['exp']).strftime('%Y-%m-%d %H:%M:%S')

            #apolog.info(f' token_required {current_user},expiry_date {expiry_date}')
            if data['clientip'] != clientip:
                flask_log.error("errorCode: -3,msg:Invalid token!")
                return jsonify({'errorCode': -3,'msg':'Invalid token!'})

            #---------------
            # Validate the validity of a token on the Redis server.
            # 从Redis中获取用户名
            redis_conn = rcPool.pool().get_connection()
            username =  redis_conn.get(token).decode('utf-8')
            #apolog.info(f" request user '{current_user}' ,username in redis is '{username}' ")
            if username is None or current_user != username:
                return jsonify({'errorCode': -3,'msg':'Token has expired or Invalid token!'})

            # 保存为全局变量
            g.user_id = username

        except jwt.ExpiredSignatureError:
            return jsonify({'errorCode': -1,'msg':'Token has expired!'})
        except jwt.InvalidTokenError:
            return jsonify({'errorCode': -1,'msg':'Invalid token!'})
        except Exception as e:
            flask_log.error(f"errorCode: -2,msg:{str(e)}")
            return jsonify({'errorCode': -2,'msg':'read token failed!'})
        finally:
            if redis_conn: 
                rcPool.pool().release_connection(redis_conn)


        return f(current_user, *args, **kwargs)

    return decorated

def after_request_decorator(f):
    """
    装饰器函数，用于在接口调用完成后调用,实现统一的日志收集
    [*user_id*] {user_id} [*remote_addr*] {request.remote_addr} [*path*] {request.path} [*request_params*] {request_data} [*response_data*] {response_data}
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        response = f(*args, **kwargs)

        # 获取请求信息
        request_data = {}
        request_data['url'] = request.url
        request_data['method'] = request.method

        # 从全局变量中获取当前用户信息
        user_id = getattr(g, 'user_id', None)

        # 获取请求参数
        if request.method == 'GET':
            request_data['params'] = dict(request.args)
        elif request.method == 'POST':
            content_type = request.headers.get('Content-Type')
            if content_type == 'application/x-www-form-urlencoded':
                request_data['params'] = dict(request.form)
            elif content_type == 'application/json':
                request_data['params'] = request.json or request.get_json()
            elif content_type.startswith('multipart/form-data'):
                request_data['params'] = {}
                for key, value in request.form.items():
                    request_data['params'][key] = value
                for key, file in request.files.items():
                    request_data['params'][key] = 'FileStorage: %s (%s)' % (file.filename, file.content_type)

        # 获取响应结果
        response_data = None
        try:
            response_data = response.data.decode('utf-8')
        # 在接口调用完成后调用日志审计记录函数
        except Exception as e:
            flask_log.error(str(e))

        flask_log.debug(f' [*user_id*] {user_id} [*remote_addr*] {request.remote_addr} [*path*] {request.path} [*request_params*] {request_data} [*response_data*] {response_data}')
        return response
    return wrapper

# 用户登录
@app.route('/login',methods=['POST'])
@after_request_decorator
def login():
    auth = request.authorization
    ##apolog.info(f'auth {auth}')

    if not auth or not auth.username or not auth.password:
       return jsonify({'errorCode': -1,'accessToken':{},'msg':'request.authorization is empty'})

    username = auth.username
    password = auth.password
    clientip = request.headers.get('X-Forwarded-For', request.remote_addr)

    rpc_proxy = pool.get_connection()
    errorCode,msg = rpc_proxy.usersService.authenticate(username,password)
    pool.put_connection(rpc_proxy)

    if errorCode !=0 :
        return jsonify({f'errorCode': errorCode,'accessToken':{},'permission':{}, 'msg':msg})

    # 调用permissionsService 获取权限信息
    rpc_proxy = pool.get_connection()
    errorCode,permission,msg = rpc_proxy.permissionsService.get_user_roles_permissions_functions(username)
    pool.put_connection(rpc_proxy)

    if errorCode !=0 :
        return jsonify({f'errorCode': errorCode,'accessToken':{},'permission':{}, 'msg':msg})

    # 生成JWT Token

    expiration_time = datetime.datetime.utcnow() + datetime.timedelta(seconds= JWT_EXPIRATION_DELTA)
    token = jwt.encode( {'username': auth.username, 'exp': expiration_time,'clientip': clientip}, SECRET_KEY)

    #--------- saved into redis ------------------# 
    redis_conn = None
    try:
        redis_conn = rcPool.pool().get_connection()
        redis_conn.set(token, username)
        redis_conn.expire(token, JWT_EXPIRATION_DELTA)
    except Exception as e:
        flask_log.error({str(e)})
    finally:
        if redis_conn:
            rcPool.pool().release_connection(redis_conn)

    return jsonify({'errorCode': errorCode,'accessToken': token, 'permission':permission, 'msg': msg})

# logout
@app.route('/logout', methods=['POST'])
@token_required
@after_request_decorator
def logout(current_user):
    token = None
    redis_conn = None
    try:
        if 'Authorization' in request.headers:
    #-------clear the token from redis------------# 
            token = request.headers['Authorization'].split(' ')[1]
            redis_conn = rcPool.pool().get_connection()
            redis_conn.delete(token)
            #jwt_redis.delete(token)
    #--------------------------------------------#
            return jsonify({'errorCode': 0,'msg': 'user logout successfully'})
    except Exception as e:
        return jsonify({'errorCode': -2,'msg': 'user logout failed'})
    finally:
        if redis_conn: 
            rcPool.pool().release_connection(redis_conn)

    return jsonify({'errorCode': -1,'msg': 'user logout failed'})

# 远程shutdown flask server
@app.route('/stopServer', methods=['GET'])
@token_required
@after_request_decorator
def stopServer(current_user):
    os.kill(os.getpid(), signal.SIGINT)
    return jsonify({"success": True,"message":"Server is shutting down..." })


#--------------------usesService----------------------------------------#
@app.route('/addUser', methods=['POST'])
@token_required
@after_request_decorator
def addUser(current_user):

    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})
    user = request.get_json()

    rpc_proxy = pool.get_connection()
    errorCode,msg = rpc_proxy.usersService.add_user(user)
    pool.put_connection(rpc_proxy)

    return jsonify({f"errorCode": errorCode,"msg": str(msg)})

@app.route('/editUser', methods=['POST'])
@token_required
@after_request_decorator
def editUser(current_user):

    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})
    user = request.get_json()

    rpc_proxy = pool.get_connection()
    errorCode,msg = rpc_proxy.usersService.edit_user(user)
    pool.put_connection(rpc_proxy)

    return jsonify({f"errorCode": errorCode,"msg": str(msg)})

@app.route('/deleteUser', methods=['POST'])
@token_required
@after_request_decorator
def deleteUser(current_user):

    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})
    user = request.get_json()

    rpc_proxy = pool.get_connection()
    errorCode,msg = rpc_proxy.usersService.delete_user(user["userId"])
    pool.put_connection(rpc_proxy)

    return jsonify({f"errorCode": errorCode,"msg": str(msg)})

# 根据组织机构代码查用户列表
@app.route('/getUserList', methods=['POST'])
@token_required
@after_request_decorator
def getUserList(current_user):


    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})
    org = request.get_json()

    rpc_proxy = pool.get_connection()
    errorCode,data,msg = rpc_proxy.usersService.get_users_by_organization_code(org['organizationCode'])
    pool.put_connection(rpc_proxy)

    return jsonify({f"errorCode": errorCode, "data":data, "msg": str(msg)})

# 根据组织机构代码查用户列表/分页
@app.route('/paginatedGetUserList', methods=['POST'])
@token_required
@after_request_decorator
def paginatedGetUserList(current_user):

    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})
    org = request.get_json()

    page = int(org['page'])
    pageSize = int(org['pageSize'])

    rpc_proxy = pool.get_connection()
    errorCode,data,msg = rpc_proxy.usersService.get_users_by_organization_code(org['organizationCode'])
    pool.put_connection(rpc_proxy)

    if errorCode == 0 :

        start = (page - 1) * pageSize
        end = start + pageSize

        try:
            new_data = arrayCopy.arrayCopyByRange(data,start,end)
            return jsonify({f"errorCode": errorCode,"data": new_data,"msg":None})
        except arrayCopyByRangeException as e:
            return jsonify({f"errorCode": -2,"data":{},"msg": str(e)})

    return jsonify({f"errorCode": errorCode,"data":result,"msg": str(msg)})

# 根据用户id查详细信息
@app.route('/getUserInformation', methods=['POST'])
@token_required
@after_request_decorator
def getUserInformation(current_user):

    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})
    user = request.get_json()

    rpc_proxy = pool.get_connection()
    errorCode,data,msg = rpc_proxy.usersService.get_user_by_id(user['userId'])
    pool.put_connection(rpc_proxy)

    return jsonify({f"errorCode": errorCode, "data":data, "msg": str(msg)})

@app.route('/resetPassword', methods=['POST'])
@after_request_decorator
def resetPassword():

    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})
    user = request.get_json()

    rpc_proxy = pool.get_connection()
    errorCode,msg = rpc_proxy.generalMessageService.reset_password_over_email(user['mail'])
    pool.put_connection(rpc_proxy)

    return jsonify({f"errorCode": errorCode,"msg": str(msg)})

@app.route('/authMailToken', methods=['POST'])
@after_request_decorator
def authMailToken():

    try:

        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(' ')[1]
            rpc_proxy = pool.get_connection()
            errorCode,data,msg = rpc_proxy.usersService.auth_mail_token(token)
            pool.put_connection(rpc_proxy)

            return jsonify({f"errorCode": errorCode,"msg": str(msg)})
        else:
            return jsonify({f"errorCode": -1,"msg": "Authorization is None"})
    except Exception as e:
        return jsonify({f"errorCode": -2,"msg": "Auth mail token failed"})


@app.route('/password-reset/<token>', methods=['GET', 'POST'])
@after_request_decorator
def resetPasswordHTML(token):

    rpc_proxy = pool.get_connection()
    errorCode,data,msg = rpc_proxy.usersService.auth_mail_token(token)
    pool.put_connection(rpc_proxy)

    if errorCode != 0:
        return jsonify({f"errorCode":errorCode,"msg" :"Auth mail token failed"})

    if request.method == 'POST':

        password = request.form['password']

        #update password where email = data['mail']
        rpc_proxy = pool.get_connection()
        errorCode,msg = rpc_proxy.usersService.reset_password_by_email(data['email'],password)
        if errorCode == 0:
            errorCode ,msg = rpc_proxy.securityService.deleteToken(token)
        pool.put_connection(rpc_proxy)

        if errorCode == 0:
            #return 'Your password has been reset.'
            return redirect('https://aiportal.apoco.com.cn/#/login')
        else:
            return 'Your password reset failed.'
        
    return render_template('reset_password.html')

# 查询roles
@app.route('/getAllRoles', methods=['GET'])
@token_required
@after_request_decorator
def getAllRoles(current_user):

    rpc_proxy = pool.get_connection()
    errorCode,data,msg = rpc_proxy.rolesService.get_all_roles()
    pool.put_connection(rpc_proxy)

    return jsonify({f"errorCode": errorCode,"data":str(data),"msg": str(msg)})

# 查询roles
@app.route('/getUserPermission', methods=['GET'])
@token_required
@after_request_decorator
def getUserPermission(current_user):

    userId = request.args.get('userId')

    rpc_proxy = pool.get_connection()
    errorCode,data,msg = rpc_proxy.permissionsService.get_user_roles_permissions_functions(userId)
    pool.put_connection(rpc_proxy)

    return jsonify({f"errorCode": errorCode,"data":str(data),"msg": str(msg)})

#-------------------organizationsService--------------------------------#
# 查询所有组织机构信息
@app.route('/getAllOrganizationsInformation', methods=['GET'])
@token_required
@after_request_decorator
def getAllOrganizationsInformation(current_user):

    rpc_proxy = pool.get_connection()
    errorCode,data,msg = rpc_proxy.organizationsService.getAllOrganizationsInformation()
    pool.put_connection(rpc_proxy)

    return jsonify({f"errorCode": errorCode,"data":str(data),"msg": str(msg)})

# 查询所有组织机构信息/分页
@app.route('/paginatedGetAllOrganizationsInformation', methods=['GET'])
@token_required
@after_request_decorator
def paginatedGetAllOrganizationsInformation(current_user):

    page = int(request.args.get('page'))
    pageSize = int(request.args.get('pageSize'))

    rpc_proxy = pool.get_connection()
    errorCode,data,msg = rpc_proxy.organizationsService.getAllOrganizationsInformation()
    pool.put_connection(rpc_proxy)

    if errorCode == 0 :

        start = (page - 1) * pageSize
        end = start + pageSize

        try:
            new_data = arrayCopy.arrayCopyByRange(data["organizations"],start,end)
            return jsonify({f"errorCode": errorCode,"data": new_data,"msg":None})
        except arrayCopyByRangeException as e:
            return jsonify({f"errorCode": -2,"data":{},"msg": str(e)})

    return jsonify({f"errorCode": errorCode,"data":result,"msg": str(msg)})

# 查询指定组织机构信息
@app.route('/getOneOrganizationInformation', methods=['GET'])
@token_required
@after_request_decorator
def getOneOrganizationInformation(current_user):

    organizationCode = request.args.get('organizationCode')
    rpc_proxy = pool.get_connection()
    errorCode,data,msg = rpc_proxy.organizationsService.getOneOrganizationInformation(organizationCode)
    pool.put_connection(rpc_proxy)

    return jsonify({f"errorCode": errorCode,"data":str(data),"msg": str(msg)})

# 增加一个组织机构的详细信息
@app.route('/addOneOrganizationInformation', methods=['POST'])
@token_required
@after_request_decorator
def addOneOrganizationInformation(current_user):

    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

    organization_json = request.get_json()

    rpc_proxy = pool.get_connection()
    errorCode,msg = rpc_proxy.organizationsService.addOneOrganizationInformation(organization_json)
    pool.put_connection(rpc_proxy)

    return jsonify({f"errorCode": errorCode,"data":str(None),"msg": str(msg)})

# 修改一个组织机构的详细信息
@app.route('/editOneOrganizationInformation', methods=['POST'])
@token_required
@after_request_decorator
def editOneOrganizationInformation(current_user):

    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

    organization_json = request.get_json()

    rpc_proxy = pool.get_connection()
    errorCode,msg = rpc_proxy.organizationsService.editOneOrganizationInformation(organization_json)
    pool.put_connection(rpc_proxy)

    return jsonify({f"errorCode": errorCode,"data":str(None),"msg": str(msg)})

# 删除一个组织机构的详细信息
@app.route('/delOneOrganizationInformation', methods=['POST'])
@token_required
@after_request_decorator
def delOneOrganizationInformation(current_user):

    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

    orgInfo_json = request.get_json()

    rpc_proxy = pool.get_connection()
    res,errMsg = rpc_proxy.organizationsService.delOneOrganizationInformation(orgInfo_json["organizationCode"])
    pool.put_connection(rpc_proxy)

    return jsonify({f"errorCode": res,"data":{},"msg": errMsg})
#End-----------------organizationsService --------------------------------#

#Begin---------------------projectsService--------------------------------#
# 查询所有组织机构下的project信息
@app.route('/getAllProjectsInformation', methods=['GET'])
@token_required
@after_request_decorator
def getAllProjectsInformation(current_user):

    rpc_proxy = pool.get_connection()
    errorCode,data,msg = rpc_proxy.projectsService.getAllProjectsInformation()
    pool.put_connection(rpc_proxy)

    return jsonify({f"errorCode": errorCode,"data":str(data),"msg": str(msg)})

# 查询指定组织机构信息
@app.route('/getOneProjectInformation', methods=['GET'])
@token_required
@after_request_decorator
def getOneProjectInformation(current_user):

    organizationCode = request.args.get('organizationCode')
    projectId = request.args.get('projectId')

    rpc_proxy = pool.get_connection()
    errorCode,data,msg = rpc_proxy.projectsService.getOneProjectInformation(organizationCode,projectId)
    pool.put_connection(rpc_proxy)

    return jsonify({f"errorCode": errorCode,"data":str(data),"msg": str(msg)})

# 为一个组织机构增加项目详细信息
@app.route('/addOneProjectInformation', methods=['POST'])
@token_required
@after_request_decorator
def addOneProjectInformation(current_user):

    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

    project_json = request.get_json()

    rpc_proxy = pool.get_connection()
    errorCode,msg = rpc_proxy.projectsService.addOneProjectInformation(project_json)
    pool.put_connection(rpc_proxy)

    return jsonify({f"errorCode": errorCode,"data":str(None),"msg": str(msg)})

# 修改某一个组织机构的某个项目详细信息
@app.route('/editOneProjectInformation', methods=['POST'])
@token_required
@after_request_decorator
def editOneProjectInformation(current_user):

    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

    project_json = request.get_json()

    rpc_proxy = pool.get_connection()
    errorCode,msg = rpc_proxy.projectsService.editOneProjectInformation(project_json)
    pool.put_connection(rpc_proxy)

    return jsonify({f"errorCode": errorCode,"data":str(None),"msg": str(msg)})

# 删除某一个组织机构的某个项目详细信息
@app.route('/delOneProjectInformation', methods=['POST'])
@token_required
@after_request_decorator
def delOneProjectInformation(current_user):

    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

    project_json = request.get_json()

    rpc_proxy = pool.get_connection()
    errorCode,msg = rpc_proxy.projectsService.delOneProjectInformation(project_json)
    pool.put_connection(rpc_proxy)

    return jsonify({f"errorCode": errorCode,"data":str(None),"msg": str(msg)})
#End-----------------------projectsService--------------------------------#

#Begin---------------------projectBuildingsService------------------------#

# 为一个组织机构下的某个项目添加楼栋详细信息
@app.route('/addOneProjectBuildingInformation', methods=['POST'])
@token_required
@after_request_decorator
def addOneProjectBuildingInformation(current_user):

    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

    building_json = request.get_json()

    rpc_proxy = pool.get_connection()
    errorCode,msg = rpc_proxy.projectBuildingsService.addOneProjectBuildingInformation(building_json)
    pool.put_connection(rpc_proxy)

    return jsonify({f"errorCode": errorCode,"data":str(None),"msg": str(msg)})

# 为一个组织机构下的某个项目修改楼栋详细信息
@app.route('/editOneProjectBuildingInformation', methods=['POST'])
@token_required
@after_request_decorator
def editOneProjectBuildingInformation(current_user):

    #apolog.info('before call editOneProjectBuildingInformation')
    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

    building_json = request.get_json()

    rpc_proxy = pool.get_connection()
    errorCode,msg = rpc_proxy.projectBuildingsService.editOneProjectBuildingInformation(building_json)
    pool.put_connection(rpc_proxy)

    return jsonify({f"errorCode": errorCode,"data":str(None),"msg": str(msg)})
#--------------------

# 删除某个项目中楼栋信息
@app.route('/deleteOneProjectBuilding', methods=['POST'])
@token_required
@after_request_decorator
def deleteOneProjectBuilding(current_user):

    #apolog.info('before call deleteOneProjectBuilding')
    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

    building_json = request.get_json()
    #apolog.info(f" {building_json} ")

    rpc_proxy = pool.get_connection()
    errorCode,msg = rpc_proxy.projectBuildingsService.deleteOneProjectBuilding(building_json)
    pool.put_connection(rpc_proxy)

    #apolog.info('called deleteOneProjectBuilding')
    return jsonify({f"errorCode": errorCode,"msg": str(msg)})

# 为一个组织机构下的某个项目的某个楼栋添加空间详细信息
@app.route('/addFloorsInformationForBuilding', methods=['POST'])
@token_required
@after_request_decorator
def addFloorsInformationForBuilding(current_user):

    #apolog.info('before call addFloorsInformationForBuilding')
    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

    buildingFloors_json = request.get_json()
    #apolog.info(f" {buildingFloors_json} ")

    rpc_proxy = pool.get_connection()
    errorCode,msg = rpc_proxy.projectBuildingsService.addFloorsInformationForBuilding(buildingFloors_json)
    pool.put_connection(rpc_proxy)

    #apolog.info('called addFloorsInformationForBuilding')
    return jsonify({f"errorCode": errorCode,"data":str(None),"msg": str(msg)})

# 编辑一个组织机构下的某个项目的某个楼栋楼层空间详细信息,可同时新增、修改多层的空间信息。
@app.route('/editFloorsInformationForBuilding', methods=['POST'])
@token_required
@after_request_decorator
def editFloorsInformationForBuilding(current_user):

    #apolog.info('before call editFloorsInformationForBuilding')
    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

    buildingFloors_json = request.get_json()
    #apolog.info(f" {buildingFloors_json} ")

    rpc_proxy = pool.get_connection()
    errorCode,msg = rpc_proxy.projectBuildingsService.editFloorsInformationForBuilding(buildingFloors_json)
    pool.put_connection(rpc_proxy)

    #apolog.info('called editFloorsInformationForBuilding')
    return jsonify({f"errorCode": errorCode,"data":str(None),"msg": str(msg)})

# 查询组织机构下的某个项目所有楼栋信息
@app.route('/getOneProjectAllBuildingsInformation', methods=['POST'])
@token_required
@after_request_decorator
def getOneProjectAllBuildingsInformation(current_user):

    #apolog.info('before call getOneProjectAllBuildingsInformation')
    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

    project_json = request.get_json()
    #apolog.info(f" {project_json} ")

    rpc_proxy = pool.get_connection()
    errorCode,data,msg = rpc_proxy.projectBuildingsService.getOneProjectAllBuildingsInformation(project_json)
    pool.put_connection(rpc_proxy)

    #apolog.info('called getOneProjectAllBuildingsInformation')
    return jsonify({f"errorCode": errorCode,"data":data,"msg": str(msg)})

#End-----------------------projectBuildingsService------------------------#


# Begin---------------------MultiJobFrameService---------------------------#
# 提交计算任务
@app.route('/submitJob', methods=['POST'])
@token_required
@after_request_decorator
def submitJob(current_user):

    #apolog.info('before call submitJob')
    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

    job_json = request.get_json()
    #apolog.info(f" {job_json} ")

    rpc_proxy = pool.get_connection()
    errorCode,msg = rpc_proxy.multiJobFrameService.submitJob(job_json)
    pool.put_connection(rpc_proxy)

    #apolog.info('called submitJob')
    return jsonify({f"errorCode": errorCode,"msg": str(msg)})

# 查询任务状态
@app.route('/getJobStatus', methods=['GET'])
@token_required
@after_request_decorator
def getJobStatus(current_user):

    #apolog.info('before call submitJob')
    job_id = request.args.get("jobId")

    rpc_proxy = pool.get_connection()
    errorCode,data,msg = rpc_proxy.multiJobFrameService.getJobStatus(job_id)
    pool.put_connection(rpc_proxy)

    #apolog.info('called getJobStatus')
    return jsonify({f"errorCode": errorCode,"data":data,"msg": str(msg)})

# 查询任务list
@app.route('/getJobList', methods=['GET'])
@token_required
@after_request_decorator
def getJobList(current_user):

    #apolog.info('before call getJobList')
    organizationCode = request.args.get("organizationCode")

    rpc_proxy = pool.get_connection()
    errorCode,data,msg = rpc_proxy.multiJobFrameService.getJobList(organizationCode)
    pool.put_connection(rpc_proxy)
    #apolog.info('called getJobList')
    return jsonify({f"errorCode": errorCode,"data":data,"msg": str(msg)})

# 查询所有组织机构信息/分页
@app.route('/paginatedGetJobList', methods=['GET'])
@token_required
@after_request_decorator
def paginatedGetJobList(current_user):

    #apolog.info('before call paginatedGetJobList')

    organizationCode = request.args.get("organizationCode")
    page = int(request.args.get('page'))
    pageSize = int(request.args.get('pageSize'))

    #errorCode,data,msg = getAllOrganizationsInformation()
    rpc_proxy = pool.get_connection()
    errorCode,data,msg = rpc_proxy.multiJobFrameService.getJobList(organizationCode)
    pool.put_connection(rpc_proxy)

    if errorCode == 0 :

        start = (page - 1) * pageSize
        end = start + pageSize

        try:
            new_data = arrayCopy.arrayCopyByRange(data,start,end)
            return jsonify({f"errorCode": errorCode,"data": new_data,"msg":None})
        except arrayCopyByRangeException as e:
            return jsonify({f"errorCode": -2,"data":{},"msg": str(e)})

# End-----------------------MultiJobFrameService---------------------------#

# 查询建筑空间类型信息列表
@app.route('/getBuildingSpaceTypeList', methods=['GET'])
@token_required
@after_request_decorator
def getBuildingSpaceTypeList(current_user):
    #apolog.info('before call getBuildingSpaceTypeList')

    rpc_proxy = pool.get_connection()
    resp = rpc_proxy.buildingSpaceTypeService.getBuildingSpaceTypeList()
    pool.put_connection(rpc_proxy)

    #apolog.info('called getBuildingSpaceTypeList')
    return jsonify({f"errorCode": 0,"data":str(resp),"msg": "message: getBuildingSpaceTypeList successfully"})

# 查询建筑类型信息列表
@app.route('/getBuildingClassificationsList', methods=['GET'])
@token_required
@after_request_decorator
def getBuildingClassificationsList(current_user):
    #apolog.info('before call getBuildingClassificationsList')

    rpc_proxy = pool.get_connection()
    resp = rpc_proxy.buildingClassificationsService.getBuildingClassificationsList()
    pool.put_connection(rpc_proxy)

    #apolog.info('called getBuildingClassificationsList')
    return jsonify({f"errorCode": 0,"data":str(resp),"msg": "message: getBuildingClassificationsList successfully"})

# 查询气候分区
@app.route('/getClimateZone', methods=['GET'])
@token_required
@after_request_decorator
def getClimateZone(current_user):
    #apolog.info('before call getgetClimateZone')

    rpc_proxy = pool.get_connection()
    resp = rpc_proxy.climateZoneService.getClimateZone()
    pool.put_connection(rpc_proxy)

    #apolog.info('called getClimateZone')
    return jsonify({f"errorCode": 0,"data":resp["climateZone"],"msg": "message: getClimateZone successfully"})

# 查询建筑结构分类
@app.route('/getBuildingStructureTypeInformation', methods=['GET'])
@token_required
@after_request_decorator
def getBuildingStructureTypeInformation(current_user):
    rpc_proxy = pool.get_connection()
    resp = rpc_proxy.BasicInformationConfigurationService.getBuildingStructureTypeInformation()
    pool.put_connection(rpc_proxy)

    #apolog.info('called getClimateZone')
    return jsonify({f"errorCode": 0,"data":resp["Structure Type"],"msg": "message: getBuildingStructureTypeInformation successfully"})

# 查询建筑结构分类
@app.route('/getOrientationInformation', methods=['GET'])
@token_required
@after_request_decorator
def getOrientationInformation(current_user):
    rpc_proxy = pool.get_connection()
    resp = rpc_proxy.BasicInformationConfigurationService.getOrientationInformation()
    pool.put_connection(rpc_proxy)

    return jsonify({f"errorCode": 0,"data":resp["Orientation"],"msg": "message: getOrientationInformation successfully"})

# 查询全部空间组成详细信息,包括如所有户型及户型的组成信息，楼梯间，公共空间，通道等
@app.route('/getAllBuildingSpaceCompositionInformation', methods=['POST'])
@token_required
@after_request_decorator
def getAllBuildingSpaceCompositionInformation(current_user):
    #apolog.info('before call getAllBuildingSpaceCompositionInformation')
    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

    projectInfo_json = request.get_json()

    rpc_proxy = pool.get_connection()
    errorCode,data,msg = rpc_proxy.buildingSpaceCompositionInformationService.getAllBuildingSpaceCompositionInformation(projectInfo_json)
    pool.put_connection(rpc_proxy)

    #apolog.info('called getAllBuildingSpaceCompositionInformation')
    return jsonify({f"errorCode": errorCode,"data":str(data),"msg": str(msg)})

# 查询指定空间组成详细信息,包括如某个户型及户型的组成信息
@app.route('/getOneBuildingSpaceCompositionInformation', methods=['POST'])
@token_required
@after_request_decorator
def getOneBuildingSpaceCompositionInformation(current_user):

    #apolog.info('before call getOneBuildingSpaceCompositionInformation')
    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

    buildingSpace_json = request.get_json()
    #apolog.info(f" {buildingSpace_json} ")

    rpc_proxy = pool.get_connection()
    errorCode,data,msg = rpc_proxy.buildingSpaceCompositionInformationService.getOneBuildingSpaceCompositionInformation(buildingSpace_json)
    pool.put_connection(rpc_proxy)

    #apolog.info('called getOneBuildingSpaceCompositionInformation')
    return jsonify({f"errorCode": errorCode,"data":str(data),"msg": str(msg)})

# 增加一项空间组成详细信息,如添加某个户型及户型的组成信息
@app.route('/addOneBuildingSpaceCompositionInformation', methods=['POST'])
@token_required
@after_request_decorator
def addOneBuildingSpaceCompositionInformation(current_user):

    #apolog.info('before call addOneBuildingSpaceCompositionInformation')
    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

    buildingSpace_json = request.get_json()
    #apolog.info(f" {buildingSpace_json} ")

    rpc_proxy = pool.get_connection()
    res,errMsg = rpc_proxy.buildingSpaceCompositionInformationService.addOneBuildingSpaceCompositionInformation(buildingSpace_json)
    pool.put_connection(rpc_proxy)

    #apolog.info('called addOneBuildingSpaceCompositionInformation')
    return jsonify({f"errorCode": res,"data":{},"msg": errMsg})

# 编辑修改一项空间组成详细信息,如某个户型及户型的组成信息
@app.route('/editOneBuildingSpaceCompositionInformation', methods=['POST'])
@token_required
@after_request_decorator
def editOneBuildingSpaceCompositionInformation(current_user):

    #apolog.info('before call editOneBuildingSpaceCompositionInformation')
    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

    buildingSpace_json = request.get_json()
    #apolog.info(f" {buildingSpace_json} ")

    rpc_proxy = pool.get_connection()
    res,errMsg = rpc_proxy.buildingSpaceCompositionInformationService.editOneBuildingSpaceCompositionInformation(buildingSpace_json)
    pool.put_connection(rpc_proxy)

    #apolog.info('called editOneBuildingSpaceCompositionInformation')
    return jsonify({f"errorCode": res,"data":{},"msg": errMsg})

# 删除一项空间组成详细信息,如某个户型及户型的组成信息
@app.route('/delOneBuildingSpaceCompositionInformation', methods=['POST'])
@token_required
@after_request_decorator
def delOneBuildingSpaceCompositionInformation(current_user):

    #apolog.info('before call delOneBuildingSpaceCompositionInformation')
    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

    buildingSpace_json = request.get_json()
    #apolog.info(f" {buildingSpace_json} ")
    rpc_proxy = pool.get_connection()
    res,errMsg = rpc_proxy.buildingSpaceCompositionInformationService.delOneBuildingSpaceCompositionInformation(buildingSpace_json)
    pool.put_connection(rpc_proxy)

    #apolog.info('called delOneBuildingSpaceCompositionInformation')
    return jsonify({f"errorCode": res,"msg": errMsg})


# 查询1 个项目1个楼栋的详细信息
@app.route('/getOneProjectBuildingInformation', methods=['POST'])
@token_required
@after_request_decorator
def getOneProjectBuildingInformation(current_user):
    #apolog.info('before call getOneProjectBuildingInformation')

#    organizationCode = request.args.get('organizationCode')
#    projectId = request.args.get('projectId')
#    buildingId = request.args.get('buildingId')
    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

    buildingInfo_json = request.get_json()

    rpc_proxy = pool.get_connection()
    errorCode,data,msg = rpc_proxy.projectBuildingsService.getOneProjectBuildingInformation(buildingInfo_json)
    pool.put_connection(rpc_proxy)

    #apolog.info('called getOneProjectBuildingInformation')
    return jsonify({f"errorCode": errorCode,"data":str(data),"msg": str(msg)})

#
# test call image_recognition_service
#@app.route('/image_recognition_service', methods=["GET","POST"])
#@token_required
#def call_service3(current_user):
#    with ClusterRpcProxy(config_mq) as rpc:
#        file = request.files['file']
#        encoded=base64.b64encode(file.read())
#        b4 =str(encoded,'utf-8')
#        result =rpc.image_recognition_service.image_recognition_service(b4)
#        return result,200

@app.route('/predict', methods=['POST'])
@token_required
@after_request_decorator
def predict(current_user):

    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

    p_json = request.get_json()

    rpc_proxy = pool.get_connection()
    errorCode,msg = rpc_proxy.AC2NNetPredicterService.predict(p_json["predict_file_name"])
    pool.put_connection(rpc_proxy)

    return jsonify({f"errorCode": errorCode,"data":{},"msg": str(msg)})

# get_predict_queue_info
@app.route('/get_predict_queue_info', methods=['GET'])
@token_required
@after_request_decorator
def get_predict_queue_info(current_user):

    rpc_proxy = pool.get_connection()
    data = rpc_proxy.AC2NNetPredicterService.get_predict_queue_info()
    pool.put_connection(rpc_proxy)

    return jsonify({f"errorCode":0,"data":data,"msg":"Successfully"})

# get_task_history
@app.route('/get_task_history', methods=['POST'])
@token_required
@after_request_decorator
def get_task_history(current_user):

    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})
    args = request.get_json()

    page = int(args['page'])
    pageSize = int(args['pageSize'])

    rpc_proxy = pool.get_connection()
    errorCode, msg = rpc_proxy.AC2NNetPredicterService.get_task_history()
    pool.put_connection(rpc_proxy)

    if errorCode == 0 :

        start = (page - 1) * pageSize
        end = start + pageSize

        try:
            new_data = arrayCopy.arrayCopyByRange(msg,start,end)
            return jsonify({f"errorCode": errorCode,"data": new_data, "rows": len(msg) })
        except arrayCopyByRangeException as e:
            return jsonify({f"errorCode": -2,"data":{}, "rows": 0 })

#    rpc_proxy = pool.get_connection()
#    errorCode, msg = rpc_proxy.AC2NNetPredicterService.get_task_history()
#    pool.put_connection(rpc_proxy)

#    return jsonify({f"errorCode": errorCode,"data": msg})
    #return jsonify(msg)

# get_predict_examples
@app.route('/get_predict_examples', methods=['POST'])
@token_required
@after_request_decorator
def get_predict_examples(current_user):

    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})
    args = request.get_json()

    page = int(args['page'])
    pageSize = int(args['pageSize'])

    rpc_proxy = pool.get_connection()
    errorCode, msg = rpc_proxy.AC2NNetPredicterService.get_predict_examples()
    pool.put_connection(rpc_proxy)

    if errorCode == 0 :

        start = (page - 1) * pageSize
        end = start + pageSize

        try:
            new_data = arrayCopy.arrayCopyByRange(msg,start,end)
            return jsonify({f"errorCode": errorCode,"data": new_data,"rows": len(msg)})
        except arrayCopyByRangeException as e:
            return jsonify({f"errorCode": -2,"data":{}, "rows": 0})

#    rpc_proxy = pool.get_connection()
#    errorCode, msg = rpc_proxy.AC2NNetPredicterService.get_predict_examples()
#    pool.put_connection(rpc_proxy)

#    return jsonify({f"errorCode": errorCode,"data": msg})

# get_predict_json_content
@app.route('/get_predict_json_content', methods=['GET'])
@token_required
@after_request_decorator
def get_predict_json_content(current_user):

    file_path = request.args.get("file_path")
    rpc_proxy = pool.get_connection()
    errorCode, msg = rpc_proxy.AC2NNetPredicterService.get_predict_json_content(file_path)
    pool.put_connection(rpc_proxy)

    return jsonify({f"errorCode": errorCode,"data": msg})
#------------------- wanghm ,begin---------------------------------
#查询获取玻璃材料数据
@app.route('/get_glass_materials', methods=['POST'])
@token_required
@after_request_decorator
def get_glass_materials(current_user):
    if not request.is_json:
        return jsonify({"errorCode": -3, "data":{} , "msg": "Missing JSON in request"})
    #column_name = request.get_json().get("column_name")
    org =request.get_json()
    rpc_proxy = pool.get_connection()
    errorCode,data,msg = rpc_proxy.get_glass_materialsService.get_glass_materials()
    pool.put_connection(rpc_proxy)
    return jsonify({f"errorCode":errorCode,"data":data,"msg":str(msg)})

#查询获取外墙材料数据
@app.route('/get_wall_materials', methods=['POST'])
@token_required
@after_request_decorator
def get_wall_materials(current_user):
    if not request.is_json:
        return jsonify({"errorCode": -3, "data": {}, "msg": "Missing JSON in request"})
    org = request.get_json()
    rpc_proxy = pool.get_connection()
    errorCode,data,msg = rpc_proxy.get_wall_materialsService.get_wall_materials()
    pool.put_connection(rpc_proxy)
    return jsonify({f"errorCode":errorCode,"data":data,"msg":str(msg)})

#查询获取窗框材料数据
@app.route('/get_wf_materials', methods=['POST'])
@token_required
@after_request_decorator
def get_wf_materials(current_user):
    if not request.is_json:
        return jsonify({"errorCode": -3, "data": {}, "msg": "Missing JSON in request"})
    org = request.get_json()
    rpc_proxy = pool.get_connection()
    errorCode,data,msg = rpc_proxy.get_wf_materialsService.get_wf_materials()
    pool.put_connection(rpc_proxy)
    return jsonify({f"errorCode":errorCode,"data":data,"msg":str(msg)})

# 增加houseTemplate
@app.route('/addHouseTemplate', methods=['POST'])
@token_required
@after_request_decorator
def addHouseTemplate(current_user):

    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

    HouseTemplate_json = request.get_json()

    rpc_proxy = pool.get_connection()
    errorCode,msg = rpc_proxy.houseTemplateService.addHouseTemplate(HouseTemplate_json)
    pool.put_connection(rpc_proxy)
    return jsonify({f"errorCode":errorCode,"msg":str(msg)})

#查询houseTemplates 列表
@app.route('/getHouseTemplates', methods=['POST'])
@token_required
@after_request_decorator
def getHouseTemplates(current_user):
    if not request.is_json:
        return jsonify({"errorCode": -3, "data": {}, "msg": "Missing JSON in request"})
    project_json = request.get_json()
    rpc_proxy = pool.get_connection()
    errorCode,data,msg = rpc_proxy.houseTemplateService.getHouseTemplates(project_json)
    pool.put_connection(rpc_proxy)
    return jsonify({f"errorCode":errorCode,"data":data,"msg":str(msg)})


#查询houseTemplatesJson
@app.route('/getHouseTemplateJson', methods=['GET'])
@token_required
@after_request_decorator
def getHouseTemplateJson(current_user):
    houseTemplateFile = request.args.get('houseTemplateFile')
    if not houseTemplateFile:
        return jsonify({"errorCode": -3, "data": {}, "msg": "Missing houseTemplateFile in request"})
    rpc_proxy = pool.get_connection()
    errorCode,data,msg = rpc_proxy.houseTemplateService.getHouseTemplateJson(houseTemplateFile)
    pool.put_connection(rpc_proxy)
    return jsonify({f"errorCode":errorCode,"data":data,"msg":str(msg)})


#---------- 2023.07.02
# 增加houseInstance
@app.route('/addHouseInstance', methods=['POST'])
@token_required
@after_request_decorator
def addHouseInstance(current_user):

    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

    HouseInstance_json = request.get_json()

    rpc_proxy = pool.get_connection()
    errorCode,msg = rpc_proxy.houseInstanceService.addHouseInstance(HouseInstance_json)
    pool.put_connection(rpc_proxy)
    return jsonify({f"errorCode":errorCode,"msg":str(msg)})

@app.route('/editHouseInstance', methods=['POST'])
@token_required
@after_request_decorator
def editHouseInstance(current_user):

    if not request.is_json:
        return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

    HouseInstance_json = request.get_json()
    data = HouseInstance_json['json_data']
    filePath = HouseInstance_json['file_path']

    rpc_proxy = pool.get_connection()
    errorCode,msg = rpc_proxy.houseInstanceService.editHouseInstance(data,filePath)
    pool.put_connection(rpc_proxy)
    return jsonify({f"errorCode":errorCode,"msg":str(msg)})

#查询houseInstances 列表
#查询houseInstances 列表
@app.route('/getHouseInstances', methods=['POST'])
@token_required
@after_request_decorator
def getHouseInstances(current_user):
    if not request.is_json:
        return jsonify({"errorCode": -3, "data": {}, "msg": "Missing JSON in request"})
    project_json = request.get_json()
    rpc_proxy = pool.get_connection()
    errorCode,data,msg = rpc_proxy.houseInstanceService.getHouseInstances(project_json)
    pool.put_connection(rpc_proxy)
    return jsonify({f"errorCode":errorCode,"data":data,"msg":str(msg)})


#查询houseInstanceJson
@app.route('/getHouseInstanceJson', methods=['GET'])
@token_required
@after_request_decorator
def getHouseInstanceJson(current_user):
    houseInstanceFile = request.args.get('houseInstanceFile')
    if not houseInstanceFile:
        return jsonify({"errorCode": -3, "data": {}, "msg": "Missing houseInstanceFile in request"})
    rpc_proxy = pool.get_connection()
    errorCode,data,msg = rpc_proxy.houseInstanceService.getHouseInstanceJson(houseInstanceFile)
    pool.put_connection(rpc_proxy)
    return jsonify({f"errorCode":errorCode,"data":data,"msg":str(msg)})

#------------------- wanghm ,end---------------------------------
@app.route('/download/<path:filename>')
#@token_required
@after_request_decorator
#def download_file(current_user,filename):
def download_file(filename):
    base_dir = "../ml_server/predicted_data"  # 文件所在的目录
    file_path = os.path.join(base_dir, filename)
    return send_file(file_path, as_attachment=True)

@after_request_decorator
def start_server(port):
    server = pywsgi.WSGIServer((iaConMg.flask_host,port), app)
    server.serve_forever()

if __name__ == '__main__':

    print("\n\n\n\n---------------------------------------------------------------------")
    print("# Apoco Intelligent Analytics Flask API Server v1.0                 #")
    print("# Requires python 3.7 +                                             #")
    print("# Copyright (c) 2023 apoco. All rights reserved.                    #")
    print("---------------------------------------------------------------------")
    print("# Server is starting ............\n")

    ports = iaConMg.flask_port
    pid_file = iaConMg.flask_pid_file
    processes = []
    # create or clear the pid file
    with open(pid_file, 'w') as f:
        f.write('')

    # write the PID of the parent process to the pid file
    parent_pid = os.getpid()

    with open(pid_file, 'a') as f:
        f.write(f'parent_pid={parent_pid}\n')


    # 启动多进程
    for port in ports:
        p = Process(target=start_server, args=(port,))
        p.start()
        processes.append(p)
        flask_log.info(f"start flask server, process_name: '{p.name}', process_pid: '{p.pid}'")

        # write the PID of each child process to the pid file
        with open(pid_file, 'a') as f:
            f.write(f"child_pid_{p.pid}={p.pid}\n")

    print("\n# Server is running ............\n")

    for p in processes:
        p.join()
