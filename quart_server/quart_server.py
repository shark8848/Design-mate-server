from nameko.standalone.rpc import ClusterRpcProxy
from datetime import timedelta
#from flask import Flask,Blueprint,render_template,request,redirect,jsonify
from flask import Flask,Blueprint,render_template,redirect
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, create_refresh_token, get_jwt_identity, get_jwt, decode_token
from gevent import pywsgi
from flask_bcrypt import Bcrypt as bcrypt
import redis
import json,os,signal
import pickle
import base64
import sys
from multiprocessing import Process
from quart import Quart, jsonify, request

sys.path.append("..")
from apocolib.apocolog4p import apoLogger as apolog

# 定义 authenticate 函数，根据用户名和密码验证用户
async def authenticate(username, password):
    user = next((user for user in users if user['username'] == username), None)
    if user and user['password'] == password:
        return user
#    if username in users and users[username] == password:
#        return username

# 初始化 Redis 客户端
redis_host = '192.168.1.19'
redis_port = 6379
redis_password = 'apoco20230213'
jwt_redis = redis.Redis(host=redis_host, port=redis_port, password=redis_password)

# MQ配置
config_mq = {'AMQP_URI': "amqp://guest:guest@192.168.1.19"}

# 初始化Flask应用程序
#app = Flask(__name__)
app = Quart(__name__)

app.config['JWT_SECRET_KEY'] = 'super-secret'  # 设置JWT密钥
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(minutes=30)  # 设置JWT有效期
app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(days=30)  # 设置JWT刷新令牌有效期

# 初始化JWTManager实例
jwt = JWTManager(app)

# 远程shutdown flask server
@app.route('/stopServer', methods=['GET'])
@jwt_required()
async def stopServer():
    os.kill(os.getpid(), signal.SIGINT)
    return jsonify({"success": True,"message":"Server is shutting down..." })


# 获取用户函数
@jwt.user_lookup_loader
async def user_lookup_callback(_jwt_header, jwt_data):
    user_id = jwt_data["sub"]
    return next((user for user in users if user['id'] == user_id), None)


# 定义一个需要身份验证的端点
@app.route('/protected')
@jwt_required()
async def protected():
    user = get_jwt_identity()
    return jsonify({'user_id': user['id'], 'username': user['username']})


# 定义登录端点
@app.route('/login', methods=['POST'])
async def login():
    """
    Login.

    Parameters
    ----------
    Arg:object
        method:post \n
        content: application/json \n
        schema: {username:?,password:?} \n
        example: {'username':'sunhy','password':'sunhy'} \n

    Returns
    -------
    200 (http)
        content: application/json \n
        schema: {errorCode:?, access_token:?, msg:?} \n
        example: {'errorCode: 0,'access_token': 'qwerewqr324324nmsfbaskjhrqjewrhkjq4hkl34h3jkh','msg': 'message:create access token successfully'} \n
                errorCode: -1, 'access_token':'','Bad username or password' \n
                errorCode: -3, 'Missing JSON in request' \n
    400/500 (http)
                errorCode: -2, 'The other error' \n
    """

    with ClusterRpcProxy(config_mq) as rpc:

        if not request.is_json:
            return jsonify({"errorCode":-3,"access_token":"","msg": "Missing JSON in request"})

        username = request.json.get('username',None)
        password = request.json.get('password',None)

        result = rpc.usersService.Authentication(request.json)

        if result['errorCode'] != 0:
            return jsonify({"errorCode":-1,"access_token":"",'msg': 'Bad username or password'})

        # 生成JWT访问令牌和刷新令牌
        access_token = create_access_token(identity=username)
        #refresh_token = create_refresh_token(identity=user['id'])
        #return jsonify({'access_token': access_token, 'refresh_token': refresh_token})

        return jsonify({f'errorCode':0,'access_token': access_token,'msg':'message:create access token successfully'})


# 定义注销端点
@app.route('/logout', methods=['DELETE'])
@jwt_required()
async def logout():
    jti = get_jwt()['jti']
    revoked_token = {'jti': jti}
    app.config['JWT_BLACKLIST_ENABLED'] = True
    app.config['JWT_BLACKLIST_TOKEN_CHECKS'] = ['access', 'refresh']
    jwt_redis.set(jti, json.dumps(revoked_token), int(app.config['JWT_ACCESS_TOKEN_EXPIRES'].total_seconds()))
    return jsonify({'msg': 'Successfully logged out'})

# 定义刷新令牌端点
@app.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
async def refresh():
    current_user = get_jwt_identity()
    new_token = create_access_token(identity=current_user)
    return jsonify({'access_token': new_token})
#
# 定义维护用户信息接口
#
#@app.route('/adduser', methods=['POST'])
#@jwt_required()
#@api.doc(description='Create a new user')
#@api.expect(user_model=api.model(apiDescription.user_model['model_name'], apiDescription.user_model['model_properties'],validate=True))
#@api.doc(responses={
#        '0': 'User added successfully',
#        '-1': 'User already exists',
#        '-2': 'unknown error',
#        '-3': 'Missing JSON in request'
#    })

#class Users(Resource):

@app.route('/adduser/', methods=['POST'])
async def adduser():
    """
    Add User


    Parameters
    ----------
    Arg:object
        method:post \n
        content: application/json \n
        schema: {username:?,password:?,organization:?,role:?} \n
        example: {'username':'sunhy','password':'sunhy','organization':'apoco','role':'admin'} \n

    Returns
    -------
    200 (http)
        content: application/json \n
        schema: {errorCode:?, data:{}, msg:?} \n
        example: {'errorCode: 0,'data': {},'msg': 'message:add user successfully'} \n
                errorCode: -1, 'User already exists' \n
                errorCode: -3, 'Missing JSON in request' \n
    400/500 (http)
                errorCode: -2, 'The other error' \n
    """
    with ClusterRpcProxy(config_mq) as rpc:
        if not request.is_json:
            return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

        apolog.info("call add user")
        errorCode, result = rpc.usersService.add_user(request.json)
        apolog.info("call add user successfully" + "errorCode " + str(errorCode) + ",msg " + result)

        return jsonify({f"errorCode": errorCode,"data":{},"msg": result})
#-------------------organizationsService--------------------------------#
# 查询所有组织机构信息
@app.route('/getAllOrganizationsInformation', methods=['GET'])
#@jwt_required() 需要jwt
async def getAllOrganizationsInformation():
    apolog.info('before call getAllOrganizationsInformation')
    with ClusterRpcProxy(config_mq) as rpc: 
        errorCode,data,msg = rpc.organizationsService.getAllOrganizationsInformation()
        apolog.info('called getAllOrganizationsInformation')
        return jsonify({f"errorCode": errorCode,"data":str(data),"msg": str(msg)})

# 查询指定组织机构信息
@app.route('/getOneOrganizationInformation', methods=['GET'])
#@jwt_required() 需要jwt
async def getOneOrganizationInformation():
    apolog.info('before call /getOneOrganizationInformation')
    with ClusterRpcProxy(config_mq) as rpc: 
        organizationCode = request.args.get('organizationCode')
        errorCode,data,msg = rpc.organizationsService.getOneOrganizationInformation(organizationCode)
        apolog.info('called /getOneOrganizationInformation')
        return jsonify({f"errorCode": errorCode,"data":str(data),"msg": str(msg)})

# 增加一个组织机构的详细信息
@app.route('/addOneOrganizationInformation', methods=['POST'])
#@jwt_required() 需要jwt
async def addOneOrganizationInformation():

    apolog.info('before call addOneOrganizationInformation')
    with ClusterRpcProxy(config_mq) as rpc: 
        if not request.is_json:
            return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

        organization_json = request.get_json()
        apolog.info(f" {organization_json} ")
        errorCode,msg = rpc.organizationsService.addOneOrganizationInformation(organization_json)
        apolog.info('called addOneOrganizationInformation')
        return jsonify({f"errorCode": errorCode,"data":str(None),"msg": str(msg)})

# 修改一个组织机构的详细信息
@app.route('/editOneOrganizationInformation', methods=['POST'])
#@jwt_required() 需要jwt
async def editOneOrganizationInformation():

    apolog.info('before call editOneOrganizationInformation')
    with ClusterRpcProxy(config_mq) as rpc: 
        if not request.is_json:
            return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

        organization_json = request.get_json()
        if organization_json is None:
            return jsonify({"errorCode": -3, "data": {}, "msg": "Missing JSON in request"})

        apolog.info(f" '{organization_json}' ")
        #errorCode,msg = await rpc.organizationsService.editOneOrganizationInformation(organization_json)
        result = rpc.organizationsService.editOneOrganizationInformation(organization_json)
        apolog.info('called editOneOrganizationInformation')
        return jsonify({f"errorCode": result[0],"data":{},"msg": result[1]})

# 删除一个组织机构的详细信息
@app.route('/delOneOrganizationInformation', methods=['POST'])
#@jwt_required() 需要jwt
async def delOneOrganizationInformation():

    apolog.info('before call delOneOrganizationInformation')
    with ClusterRpcProxy(config_mq) as rpc: 
        if not request.is_json:
            return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

        orgInfo_json = request.get_json()
        apolog.info(f" {orgInfo_json} ")
        res,errMsg = rpc.organizationsService.delOneOrganizationInformation(orgInfo_json["organizationCode"])
        apolog.info('called delOneOrganizationInformation')
        return jsonify({f"errorCode": res,"data":{},"msg": errMsg})
#End-----------------organizationsService --------------------------------#

#Begin---------------------projectsService--------------------------------#
# 查询所有组织机构下的project信息
@app.route('/getAllProjectsInformation', methods=['GET'])
#@jwt_required() 需要jwt
async def getAllProjectsInformation():
    apolog.info('before call getAllProjectsInformation')
    with ClusterRpcProxy(config_mq) as rpc: 
        errorCode,data,msg = rpc.projectsService.getAllProjectsInformation()
        apolog.info('called getAllProjectsInformation')
        return jsonify({f"errorCode": errorCode,"data":str(data),"msg": str(msg)})

# 查询指定组织机构信息
@app.route('/getOneProjectInformation', methods=['GET'])
#@jwt_required() 需要jwt
async def getOneProjectInformation():
    apolog.info('before call getOneProjectInformation')
    with ClusterRpcProxy(config_mq) as rpc: 
        organizationCode = request.args.get('organizationCode')
        projectId = request.args.get('projectId')
        errorCode,data,msg = rpc.projectsService.getOneProjectInformation(organizationCode,projectId)
        apolog.info('called getOneProjectInformation')
        return jsonify({f"errorCode": errorCode,"data":str(data),"msg": str(msg)})

# 为一个组织机构增加项目详细信息
@app.route('/addOneProjectInformation', methods=['POST'])
#@jwt_required() 需要jwt
async def addOneProjectInformation():

    apolog.info('before call addOneProjectInformation')
    with ClusterRpcProxy(config_mq) as rpc: 
        if not request.is_json:
            return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

        project_json = request.get_json()
        apolog.info(f" {project_json} ")
        errorCode,msg = rpc.projectsService.addOneProjectInformation(project_json)
        apolog.info('called addOneProjectInformation')
        return jsonify({f"errorCode": errorCode,"data":str(None),"msg": str(msg)})

# 修改某一个组织机构的某个项目详细信息
@app.route('/editOneProjectInformation', methods=['POST'])
#@jwt_required() 需要jwt
async def editOneProjectInformation():

    apolog.info('before call editOneProjectInformation')
    with ClusterRpcProxy(config_mq) as rpc: 
        if not request.is_json:
            return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

        project_json = request.get_json()
        apolog.info(f" {project_json} ")
        errorCode,msg = rpc.projectsService.editOneProjectInformation(project_json)
        apolog.info('called editOneProjectInformation')
        return jsonify({f"errorCode": errorCode,"data":str(None),"msg": str(msg)})

# 删除某一个组织机构的某个项目详细信息
@app.route('/delOneProjectInformation', methods=['POST'])
#@jwt_required() 需要jwt
async def delOneProjectInformation():

    apolog.info('before call delOneProjectInformation')
    with ClusterRpcProxy(config_mq) as rpc: 
        if not request.is_json:
            return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

        project_json = request.get_json()
        apolog.info(f" {project_json} ")
        errorCode,msg = rpc.projectsService.delOneProjectInformation(project_json)
        apolog.info('called delOneProjectInformation')
        return jsonify({f"errorCode": errorCode,"data":str(None),"msg": str(msg)})
#End-----------------------projectsService--------------------------------#

#Begin---------------------projectBuildingsService------------------------#

# 为一个组织机构下的某个项目添加楼栋详细信息
@app.route('/addOneProjectBuildingInformation', methods=['POST'])
#@jwt_required() 需要jwt
async def addOneProjectBuildingInformation():

    apolog.info('before call addOneProjectBuildingInformation')
    with ClusterRpcProxy(config_mq) as rpc: 
        if not request.is_json:
            return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

        building_json = request.get_json()
        apolog.info(f" {building_json} ")
        errorCode,msg = rpc.projectBuildingsService.addOneProjectBuildingInformation(building_json)
        apolog.info('called addOneProjectBuildingInformation')
        return jsonify({f"errorCode": errorCode,"data":str(None),"msg": str(msg)})

# 为一个组织机构下的某个项目的某个楼栋添加空间详细信息
@app.route('/addFloorsInformationForBuilding', methods=['POST'])
#@jwt_required() 需要jwt
async def addFloorsInformationForBuilding():

    apolog.info('before call addFloorsInformationForBuilding')
    with ClusterRpcProxy(config_mq) as rpc: 
        if not request.is_json:
            return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

        buildingFloors_json = request.get_json()
        apolog.info(f" {buildingFloors_json} ")
        errorCode,msg = rpc.projectBuildingsService.addFloorsInformationForBuilding(buildingFloors_json)
        apolog.info('called addFloorsInformationForBuilding')
        return jsonify({f"errorCode": errorCode,"data":str(None),"msg": str(msg)})

# 编辑一个组织机构下的某个项目的某个楼栋楼层空间详细信息,可同时新增、修改多层的空间信息。
@app.route('/editFloorsInformationForBuilding', methods=['POST'])
#@jwt_required() 需要jwt
async def editFloorsInformationForBuilding():

    apolog.info('before call editFloorsInformationForBuilding')
    with ClusterRpcProxy(config_mq) as rpc: 
        if not request.is_json:
            return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

        buildingFloors_json = request.get_json()
        apolog.info(f" {buildingFloors_json} ")
        errorCode,msg = rpc.projectBuildingsService.editFloorsInformationForBuilding(buildingFloors_json)
        apolog.info('called editFloorsInformationForBuilding')
        return jsonify({f"errorCode": errorCode,"data":str(None),"msg": str(msg)})
#End-----------------------projectBuildingsService------------------------#
# 查询建筑空间类型信息列表
@app.route('/getBuildingSpaceTypeList', methods=['GET'])
#@jwt_required() 需要jwt
async def getBuildingSpaceTypeList():
    apolog.info('before call getBuildingSpaceTypeList')
    with ClusterRpcProxy(config_mq) as rpc: 
        resp = rpc.buildingSpaceTypeService.getBuildingSpaceTypeList()
        apolog.info('called getBuildingSpaceTypeList')
        return jsonify({f"errorCode": 0,"data":str(resp),"msg": "message: getBuildingSpaceTypeList successfully"})

# 查询建筑类型信息列表
@app.route('/getBuildingClassificationsList', methods=['GET'])
#@jwt_required() 需要jwt
async def getBuildingClassificationsList():
    apolog.info('before call getBuildingClassificationsList')
    with ClusterRpcProxy(config_mq) as rpc: 
        resp = rpc.buildingClassificationsService.getBuildingClassificationsList()
        apolog.info('called getBuildingClassificationsList')
        return jsonify({f"errorCode": 0,"data":str(resp),"msg": "message: getBuildingClassificationsList successfully"})

# 查询全部空间组成详细信息,包括如所有户型及户型的组成信息，楼梯间，公共空间，通道等
@app.route('/getAllBuildingSpaceCompositionInformation', methods=['GET'])
#@jwt_required() 需要jwt
async def getAllBuildingSpaceCompositionInformation():
    apolog.info('before call getAllBuildingSpaceCompositionInformation')
    with ClusterRpcProxy(config_mq) as rpc: 
        errorCode,data,msg = rpc.buildingSpaceCompositionInformationService.getAllBuildingSpaceCompositionInformation()
        apolog.info('called getAllBuildingSpaceCompositionInformation')
        return jsonify({f"errorCode": errorCode,"data":str(data),"msg": str(msg)})

# 查询指定空间组成详细信息,包括如某个户型及户型的组成信息
@app.route('/getOneBuildingSpaceCompositionInformation', methods=['GET'])
#@jwt_required() 需要jwt
async def getOneBuildingSpaceCompositionInformation():
    apolog.info('before call getOneBuildingSpaceCompositionInformation')
    with ClusterRpcProxy(config_mq) as rpc: 
        buildingSpaceId = request.args.get('buildingSpaceId')
        errorCode,data,msg = rpc.buildingSpaceCompositionInformationService.getOneBuildingSpaceCompositionInformation(buildingSpaceId)
        apolog.info('called getOneBuildingSpaceCompositionInformation')
        return jsonify({f"errorCode": errorCode,"data":str(data),"msg": str(msg)})

# 增加一项空间组成详细信息,如添加某个户型及户型的组成信息
@app.route('/addOneBuildingSpaceCompositionInformation', methods=['POST'])
#@jwt_required() 需要jwt
async def addOneBuildingSpaceCompositionInformation():

    apolog.info('before call addOneBuildingSpaceCompositionInformation')
    with ClusterRpcProxy(config_mq) as rpc: 
        if not request.is_json:
            return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

        buildingSpace_json = request.get_json()
        apolog.info(f" {buildingSpace_json} ")
        #buildingSpaceId = request.args.get('buildingSpaceId')
        res,errMsg = rpc.buildingSpaceCompositionInformationService.addOneBuildingSpaceCompositionInformation(buildingSpace_json)
        apolog.info('called addOneBuildingSpaceCompositionInformation')
        return jsonify({f"errorCode": res,"data":{},"msg": errMsg})

# 编辑修改一项空间组成详细信息,如某个户型及户型的组成信息
@app.route('/editOneBuildingSpaceCompositionInformation', methods=['POST'])
#@jwt_required() 需要jwt
async def editOneBuildingSpaceCompositionInformation():

    apolog.info('before call editOneBuildingSpaceCompositionInformation')
    with ClusterRpcProxy(config_mq) as rpc: 
        if not request.is_json:
            return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

        buildingSpace_json = request.get_json()
        apolog.info(f" {buildingSpace_json} ")
        #buildingSpaceId = request.args.get('buildingSpaceId')
        res,errMsg = rpc.buildingSpaceCompositionInformationService.editOneBuildingSpaceCompositionInformation(buildingSpace_json)
        apolog.info('called editOneBuildingSpaceCompositionInformation')
        return jsonify({f"errorCode": res,"data":{},"msg": errMsg})

# 删除一项空间组成详细信息,如某个户型及户型的组成信息
@app.route('/delOneBuildingSpaceCompositionInformation', methods=['POST'])
#@jwt_required() 需要jwt
async def delOneBuildingSpaceCompositionInformation():

    apolog.info('before call delOneBuildingSpaceCompositionInformation')
    with ClusterRpcProxy(config_mq) as rpc: 
        if not request.is_json:
            return jsonify({"errorCode": -3,"data":{},"msg": "Missing JSON in request"})

        buildingSpace_json = request.get_json()
        apolog.info(f" {buildingSpace_json} ")
        res,errMsg = rpc.buildingSpaceCompositionInformationService.delOneBuildingSpaceCompositionInformation(buildingSpace_json["buildingSpaceId"])
        apolog.info('called delOneBuildingSpaceCompositionInformation')
        return jsonify({f"errorCode": res,"data":{},"msg": errMsg})


# 查询1 个项目1个楼栋的详细信息
@app.route('/getOneProjectBuildingInformation', methods=['GET'])
#@jwt_required() 需要jwt
async def getOneProjectBuildingInformation():
    apolog.info('before call getOneProjectBuildingInformation')
    with ClusterRpcProxy(config_mq) as rpc: 
        organizationCode = request.args.get('organizationCode')
        projectId = request.args.get('projectId')
        buildingId = request.args.get('buildingId')
        errorCode,data,msg = rpc.projectBuildingsService.getOneProjectBuildingInformation(organizationCode,projectId,buildingId)
        apolog.info('called getOneProjectBuildingInformation')
        return jsonify({f"errorCode": errorCode,"data":str(data),"msg": str(msg)})

#
# test call image_recognition_service
@app.route('/image_recognition_service', methods=["GET","POST"])
async def call_service3():
    apolog.info('before call image_recognition_service')
    with ClusterRpcProxy(config_mq) as rpc:
        apolog.info('call image_recognition_service')
        file = request.files['file']
        #img_bytes = file.read()
        apolog.info('image read')
        #encoded=base64.b64encode(file.read()).decode()
        encoded=base64.b64encode(file.read())
        b4 =str(encoded,'utf-8')
        #b4 = 'data:image/png;base64,%encoded' %encoded
#        img = "data:image/{ext};base64,{data}".format(ext=file, data=encoded)
        apolog.info('image encoded')
        #result =rpc.image_recognition_service.image_recognition_service(img_bytes)
        result =rpc.image_recognition_service.image_recognition_service(b4)
        apolog.info('called image_recognition_service')

        return result,200
#

def start_server(port):
#    app.run(port=port)
    app.run(host='10.8.0.181',port=port,debug=True)
#    server = pywsgi.WSGIServer(('10.8.0.181',port), app)
#    server.serve_forever()
# 运行应用程序
#if __name__ == '__main__':
    #app.run()
#debug 模式
#    app.run(host='10.8.0.181',port='5000',debug=True)

#    server = pywsgi.WSGIServer(('10.8.0.181',5000), app)
#    server.serve_forever()


if __name__ == '__main__':
    ports = [5001, 5002, 5003] # 端口号列表
    processes = []

    for port in ports:
        p = Process(target=start_server, args=(port,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
