# -*- coding: utf-8 -*-
"""
Program: houseInstanceService.py
Description: 提供houseinstance管理 的微服务(包括查询、新增、修改等服务)，供flask api 接口调用.
Author: Sunhy
Version: 1.0
Date: 2023-07-01
"""

from ml_server.PolyRegressor import PolyRegressor
from ml_server.MaterialWarehouse import *
from ml_server.BuildingSpaceBase import *
from apocolib import sqliteSession as sqlSession
from apocolib.NamekoLogger import namekoLogger as nameko_logger
from nameko.rpc import rpc, RpcProxy
import json
import sys
import jsonschema
import verifyJsonFileLoader
from houseModel import HouseInstance
from datetime import datetime
import pdb
import copy
# sys.path.append("..")
import sys
import os
# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将 nameko_server 目录路径添加到模块搜索路径中
sys.path.append(current_dir)

# sys.path.pop()
# from ml_server.SpaceStandard import SpaceStandard
wimWarehouse = WallInsulationMaterialWarehouse()
gmWarehouse = GlassMaterialWarehouse()
wfmWarehouse = WindowFrameMaterialWarehouse()

_ss = SpaceStandard()
pr = PolyRegressor()


class houseInstanceService:

    name = "houseInstanceService"
    houseInstance_schema = "../json/json_schema/houseInstanceSchema.json"
    loader = None

    def __init__(self):

        self.houseInstanceLoader = verifyJsonFileLoader.verifyJsonFileLoader(None,
                                                                             self.houseInstance_schema)

    def instance_to_house(self, houseInstance):
        house = House(name=houseInstance['houseInstance']['instanceName'],
                      height=houseInstance['houseInstance']['floorHeight'])

        rooms = houseInstance['houseInstance']['rooms']
        o_Rooms = []
        for room in rooms:
            o_Room = None

            if room['room_height'] != house.get_height():
                return -1, 'The room_height data is invalid. The room_height must be equal to the house_height'

            o_Room = Room(name=room['room_name'], type=room['room_type'], length=room['room_length'], width=room['room_width'], height=room['room_height'], walls=([None]) * 4,
                          area=round(room['room_length']*room['room_width'], 4))
            # o_Walls = []
            walls = room['walls']

            for wall in walls:
                o_wall = None
                print("wall_height ", wall['wall_height'])
                o_wall = Wall(
                    width=wall['wall_width'],
                    height=wall['wall_height'],
                    thickness=0.0,  # 生成一个墙的宽度,高度,厚度
                    orientation=wall['orientation'],  # 生成一个墙的朝向
                    window=None  # 生成一个墙的窗户
                )
                # 校验墙和房间的尺寸匹配问题
                try:
                    o_Room.check_wall_(o_wall.get_orientation(), o_wall)
                except Room.RoomDataError as e:
                    return -1, str(e)

                # material = wall['materials']
                m_key = wall['wall_material']['key']
                wi_material = WallInsulation(area=o_wall.get_area(
                ), warehouse=wimWarehouse, material_type=wimWarehouse.get_material(m_key))
                o_wall.set_material(wi_material)

                window = None
                try:
                    window = wall['window']
                except KeyError:
                    window = None

                o_window = None
                orientation = 4
                print("window ", window)
                # schema 中，window 为必须属性，但是可以为 {}，所以以字典是否为空进行判断.
                if isinstance(window, dict) and bool(window):
                    o_window = Window(
                        width=window['window_width'],
                        height=window['window_height'],
                        orientation=window['orientation'],
                        window_frame=None,
                        glass=None
                    )

                    if o_window.get_area() > o_wall.get_area():
                        raise Exception(
                            'The window_area exceeds the wall_area')
                    elif o_window.get_area()/o_wall.get_area() > _ss.get_window_wall_ratio_limit(o_window.get_orientation()):
                        raise Exception(
                            f"The window-to-wall ratio exceeds the specified limit. {window['window_width']} - { window['window_height'] } ")

                    if o_window.get_width() > o_wall.get_width():
                        raise Exception(
                            'The window_width exceeds the wall_width')
                    if o_window.get_height() > o_wall.get_height():
                        raise Exception(
                            'The window_height exceeds the wall_height')

                    # 使用pr 预测合理的窗框与窗户的面积比
                    wfa_ratio = round(pr.predict(o_window.get_area()), 4)

                    wf_area = round(o_window.get_area()*wfa_ratio, 4)
                    glass_area = round(o_window.get_area()-wf_area)

                    g_key = window['glass_material']['key']
                    glass = Glass(area=glass_area, warehouse=gmWarehouse,
                                  material_type=gmWarehouse.get_material(g_key))
                    o_window.set_glass(glass)

                    f_key = window['wf_material']['key']
                    window_frame = WindowFrame(
                        area=wf_area, warehouse=wfmWarehouse, material_type=wfmWarehouse.get_material(f_key))
                    o_window.set_window_frame(window_frame)

                    o_wall.add_window(copy.deepcopy(o_window))

                o_Room.add_wall(wall['orientation'], copy.deepcopy(o_wall))
            o_Rooms.append(copy.deepcopy(o_Room))

        house.set_rooms(o_Rooms)
        return house

    # edit instance
    @rpc
    def editHouseInstance(self, houseInstance, filePath):  # json

        print("houseInstance ", houseInstance)
        # 使用模板校验json 格式是否有效
        res, msg = self.houseInstanceLoader.jsonDataIsValid(houseInstance)
        if res != 0:
            return res, msg

        house = self.instance_to_house(houseInstance)
        try:
            house.save_to_json(filePath)
            house.save_to_json_cn(self.generate_chinese_filename(filePath))
            return 0, 'editHouseInstance successfully'
        except Exception as e:
            return -1, f'saveHouseInstanceToJsonFile faild. {str(e)}'

    def generate_chinese_filename(self, filename):
        # 获取文件名和扩展名
        name, ext = os.path.splitext(filename)
        # 添加 '_cn' 后缀
        chinese_filename = name + '_cn' + ext
        return chinese_filename

    # 新增一个instance
    @rpc
    def addHouseInstance(self, houseInstance):  # json

        print("houseInstance ", houseInstance)
        # 使用模板校验json 格式是否有效
        res, msg = self.houseInstanceLoader.jsonDataIsValid(houseInstance)
        if res != 0:
            return res, msg

        # pdb.set_trace()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        f_path = f'../json/instances/hins_{timestamp}.json'
        f_cn_path = f'../json/instances/hins_{timestamp}_cn.json'

        house = self.instance_to_house(houseInstance)

        # ------------------------------------------------------------------
        '''
        house = House(name=houseInstance['houseInstance']['instanceName'], height= houseInstance['houseInstance']['floorHeight']) 

        rooms = houseInstance['houseInstance']['rooms']
        o_Rooms = []
        for room in rooms:
            o_Room = None

            if room['room_height'] != house.get_height():
                return -1,'The room_height data is invalid. The room_height must be equal to the house_height'

            o_Room = Room(name=room['room_name'], type = room['room_type'], length = room['room_length'], width = room['room_width'], height = room['room_height'], walls = ([None]) * 4,
                    area=round(room['room_length']*room['room_width'],4))
            #o_Walls = []
            walls = room['walls']

            for wall in walls:
                o_wall = None
                print("wall_height ",wall['wall_height'])
                o_wall = Wall( 
                    width = wall['wall_width'], 
                    height= wall['wall_height'],
                    thickness=0.0, # 生成一个墙的宽度,高度,厚度 
                    orientation = wall['orientation'], # 生成一个墙的朝向
                    window = None # 生成一个墙的窗户
                )
                # 校验墙和房间的尺寸匹配问题
                try:
                    o_Room.check_wall_(o_wall.get_orientation(),o_wall)
                except Room.RoomDataError as e:
                    return -1,str(e)

                #material = wall['materials']
                m_key = wall['wall_material']['key']
                wi_material = WallInsulation(warehouse = wimWarehouse, material_type = wimWarehouse.get_material(m_key))
                o_wall.set_material(wi_material)

                window = None
                try:
                    window = wall['window']
                except KeyError:
                    window = None

                o_window = None
                orientation = 4
                print("window ",window)
                if isinstance(window, dict) and bool(window): # schema 中，window 为必须属性，但是可以为 {}，所以以字典是否为空进行判断.
                    o_window = Window(
                        width = window['window_width'],
                        height = window['window_height'],
                        orientation = window['orientation'],
                        window_frame = None,
                        glass = None
                    )

                    if o_window.get_area() > o_wall.get_area():
                        return -1,'The window_area exceeds the wall_area'
                    elif o_window.get_area()/o_wall.get_area() > _ss.get_window_wall_ratio_limit(o_window.get_orientation()):
                        return -1,f"The window-to-wall ratio exceeds the specified limit. {window['window_width']} - { window['window_height'] } "

                    if o_window.get_width() > o_wall.get_width():
                        return -1,'The window_width exceeds the wall_width'
                    if o_window.get_height() > o_wall.get_height():
                        return -1,'The window_height exceeds the wall_height'

                    # 使用pr 预测合理的窗框与窗户的面积比
                    wfa_ratio = round(pr.predict(o_window.get_area()),4)

                    wf_area = round(o_window.get_area()*wfa_ratio,4)
                    glass_area = round(o_window.get_area()-wf_area)

                    g_key = window['glass_material']['key']
                    glass = Glass(area = glass_area, warehouse = gmWarehouse,material_type=gmWarehouse.get_material(g_key))
                    o_window.set_glass(glass)

                    f_key = window['wf_material']['key']
                    window_frame = WindowFrame(area = wf_area, warehouse = wfmWarehouse,material_type=wfmWarehouse.get_material(f_key))
                    o_window.set_window_frame(window_frame)

                    o_wall.add_window(copy.deepcopy(o_window))
                
                o_Room.add_wall(wall['orientation'],copy.deepcopy(o_wall))
            o_Rooms.append(copy.deepcopy(o_Room))

        house.set_rooms(o_Rooms)
        '''
        # h_json = house.to_json()
        # res = self.houseInstanceLoader.dumpJsonFile(f_path,h_json)
        # if res == 0:
        try:
            house.save_to_json(f_path)
            house.save_to_json_cn(f_cn_path)
            return self.registerHouseInstance(houseInstance, f_path)
        except Exception as e:
            return -1, f'saveHouseInstanceToJsonFile faild. {str(e)}'

    def registerHouseInstance(self, houseInstance, f_path):
        # pdb.set_trace()
        nameko_logger.info(houseInstance)
        instanceName = None
        try:
            organizationCode = houseInstance['organizationCode']
            projectId = houseInstance['projectId']
            buildingId = houseInstance['buildingId']
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            instanceId = organizationCode + projectId + buildingId + timestamp
            instanceName = houseInstance["houseInstance"]["instanceName"]
            templateId = houseInstance["houseInstance"]["templateId"]
            instanceJson = f_path
            status = 1

            with sqlSession.sqliteSession().getSession() as session:

                h_ins = HouseInstance(
                    instanceId=instanceId,
                    organizationCode=organizationCode,
                    projectId=projectId,
                    buildingId=buildingId,
                    instanceName=instanceName,
                    templateId=templateId,
                    instanceJson=instanceJson,
                    status=status
                )
                session.add(h_ins)
                session.commit()
                return 0, f'add houseInstance {instanceName} successfully'

        except Exception as e:
            nameko_logger.error(
                f'add houseInstance {instanceName} error.{str(e)}')
            return -1, f'add houseInstance {instanceName} failed: {str(e)}'
        return -1, 'add houseInstance failed'

    # 查询模板列表
    @rpc
    def getHouseInstances(self, projectInfo):
        # pdb.set_trace()
        # ORG && PROJ {"organizationCode":organizationCode,"projectId":projectId,"buildingId": buildingId}
        organizationCode = None
        nameko_logger.info(projectInfo)
        try:
            organizationCode = projectInfo["organizationCode"]
            projectId = projectInfo["projectId"]
            buildingId = projectInfo["buildingId"]

            with sqlSession.sqliteSession().getSession() as session:
                houseInstances = []

                query = session.query(HouseInstance).filter(
                    HouseInstance.organizationCode == organizationCode,
                    HouseInstance.status == 1
                )

                if not projectId:
                    # 查询指定 organizationId 的记录
                    houseInstances = query.all()
                elif projectId and not buildingId:
                    # 查询指定 projectId 下的所有记录
                    houseInstances = query.filter(
                        HouseInstance.projectId == projectId
                    ).all()
                else:
                    # 查询指定 projectId 和 buildingId 的记录
                    houseInstances = query.filter(
                        HouseInstance.projectId == projectId,
                        HouseInstance.buildingId == buildingId
                    ).all()

                if houseInstances is None:
                    return -1, [], 'get_houseInstances {organizationCode} ,no data found'
                else:
                    return 0, [{
                        'instanceId': houseInstance.instanceId,
                        'organizationCode': houseInstance.organizationCode,
                        'projectId': houseInstance.projectId,
                        'buildingId': houseInstance.buildingId,
                        'instanceName': houseInstance.instanceName,
                        'templateId': houseInstance.templateId,
                        'instanceJson': houseInstance.instanceJson,
                        'status': houseInstance.status
                    } for houseInstance in houseInstances], f'get_HouseInstances {organizationCode} successfully'
        except Exception as e:
            nameko_logger.error(
                f'get_HouseInstances {organizationCode} error.{str(e)}')
            return -1, [], f'get_HouseInstances {organizationCode} error.{str(e)}'

        return -1, [], 'get_HouseInstances failed'

    # 查询某一个模板内容
    @rpc
    def getHouseInstanceJson(self, houseInstanceFile):
        try:
            with open(houseInstanceFile, 'r') as file:
                houseInstance = json.load(file)
                return 0, houseInstance, 'getHouseInstanceJson successfully'
        except Exception as e:
            return -1, {}, f'getHouseInstanceJson failed, error: {str(e)}'

    @rpc
    def getHouseInstanceJsonByInstanceId(self, instanceId):
        # CREATE TABLE house_instance (
        # instanceId varchar (100) NOT NULL UNIQUE,
        # organizationCode varchar (20) NOT NULL,
        # projectId varchar (20) NOT NULL,
        # buildingId varchar (20) NOT NULL,
        # templateId varchar (100) NOT NULL,
        # instanceName varchar (20) NOT NULL,
        # instanceJson varchar (100) NOT NULL UNIQUE,
        # status INTEGER NOT NULL DEFAULT (1),
        # PRIMARY KEY (instanceId));
        try:
            with sqlSession.sqliteSession().getSession() as session:
                houseInstance = session.query(HouseInstance).filter(
                    HouseInstance.instanceId == instanceId,
                    HouseInstance.status == 1
                ).first()
                if houseInstance is None:
                    return -1, {}, f'getHouseInstanceJsonByInstanceId {instanceId} no data found'
                else:
                    return 0, houseInstance.instanceJson, f'getHouseInstanceJsonByInstanceId {instanceId} successfully'
        except Exception as e:
            nameko_logger.error(
                f'getHouseInstanceJsonByInstanceId {instanceId} error.{str(e)}')
            return -1, {}, f'getHouseInstanceJsonByInstanceId {instanceId} error.{str(e)}'

        return -2, {}, f'getHouseInstanceJsonByInstanceId {instanceId} failed'

    '''
    @rpc
    def getHouseInstanceJson(self,houseInstanceFile):

        houseInstance  = self.houseInstanceLoader.loadJsonFile(houseInstanceFile)
        if houseInstance is not None:
            return 0,houseInstance,'getHouseInstanceJson successfully'
        else:
            return -1,{}, f'getHouseInstanceJson failed ,file {houseInstanceFile} data is none'
            '''
