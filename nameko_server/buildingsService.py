# bulidingsService.py

from apocolib import verifyJsonFileLoader
from apocolib import jsonDataValidator
from BuildingModel import Building
from apocolib import sqliteSession as sqlSession
from apocolib.NamekoLogger import namekoLogger as nameko_logger
from nameko.rpc import rpc
from datetime import datetime
import hashlib
import sys
sys.path.append("..")


class buildingsService:

    name = 'buildingsService'
    Building_schema = "./json/json_schema/projectBuildingsSchema.json"
    buildingValidator = None

    def __init__(self):
        self.buildingValidator = jsonDataValidator.jsonDataValidator(
            self.Building_schema)
        self.buildingLoader = verifyJsonFileLoader.verifyJsonFileLoader(
            None, self.Building_schema)

    @rpc
    def createBuildingInstance(self, buildingInfo):
        # 校验输入json data 是否符合
        res, msg = self.buildingValidator.jsonDataIsValid(buildingInfo)
        nameko_logger.info(f"return '{res}' msg '{msg}'")
        if res != 0:
            return res, msg

        organizationCode = buildingInfo["organizationCode"]
        projectId = buildingInfo["projectId"]
        buildingId = hashlib.md5(
            str(datetime.now()).encode('utf-8')).hexdigest()[8:-8]
        buildingInstanceDescFileName = "./json/BUIIDING-INSTANCE-DESC-{}-{}-{}.json".format(
            organizationCode, projectId, buildingId)

        try:
            res = self.pbLoader.dumpJsonFile(
                buildingInstanceDescFileName, buildingInfo)
            if res == 0:
                return res, f"insert object into '{buildingInstanceDescFileName}' successfully"
            else:
                nameko_logger.info(
                    f" insert object into '{buildingInstanceDescFileName}' failed, '{msg}' ")
                return res, msg
        except AttributeError as e:
            nameko_logger.error(
                f"Error occurred while trying to insert an object in the json file. '{buildingInstanceDescFileName}','{str(e)}'")
            return -1, f"Error occurred while trying to insert an object in the json file. '{buildingInstanceDescFileName}'"
        except Exception as e:
            nameko_logger.error(
                f"Error occurred while trying to insert an object in the json file. '{buildingInstanceDescFileName}','{str(e)}'")
            return -1, f"Error occurred while trying to insert an object in the json file. '{buildingInstanceDescFileName}'"

        return -2, f"createBuildingInstance failed"

    # def create_buildingInstanceJson(self,buildingInfo):

    @rpc
    # building_id, organization_id, project_id, building_alias, building_desc_file, building_model_file
    def add_building(self, buildingInfo):
        try:
            # 根据时间戳生成 20位的building_id
            building_id = hashlib.md5(
                str(datetime.now()).encode('utf-8')).hexdigest()[8:-8]
            organization_id = buildingInfo['organization_id']
            project_id = buildingInfo['project_id']
            building_alias = buildingInfo['building_alias']
            building_desc_file = buildingInfo['building_desc_file']
            building_model_file = buildingInfo['building_model_file']

            assert isinstance(
                organization_id, str), f'organization_id must be str, but got {type(organization_id)}'
            assert isinstance(
                project_id, str), f'project_id must be str, but got {type(project_id)}'
            assert isinstance(
                building_alias, str), f'building_alias must be str, but got {type(building_alias)}'
            assert isinstance(
                building_desc_file, str), f'building_desc_file must be str, but got {type(building_desc_file)}'
            assert isinstance(
                building_model_file, str), f'building_model_file must be str, but got {type(building_model_file)}'

            with sqlSession.sqliteSession().getSession() as session:
                new_building = Building(
                    building_id=building_id,
                    organization_id=organization_id,
                    project_id=project_id,
                    building_alias=building_alias,
                    building_desc_file=building_desc_file,
                    building_model_file=building_model_file,
                    create_time=datetime.now()
                )
                session.add(new_building)
                session.commit()
                return 0, f'add new_building {new_building} successfully'

        except Exception as e:
            nameko_logger.error(
                f'add new_building  {buildingInfo} error.{str(e)}')
            return -1, f'add new_building {buildingInfo} failed: {str(e)}'

    @rpc
    def get_building(self, building_id):
        try:
            with sqlSession.sqliteSession().getSession() as session:
                building = session.query(Building).filter(
                    Building.building_id == building_id).first()
                return 0, building

        except Exception as e:
            nameko_logger.error(f'get building {building_id} error.{str(e)}')
            return -1, f'get building {building_id} failed: {str(e)}'

    @rpc
    def get_buildings(self, organization_id, project_id):
        try:
            with sqlSession.sqliteSession().getSession() as session:
                buildings = session.query(Building).filter(
                    Building.organization_id == organization_id, Building.project_id == project_id).all()
                return 0, buildings

        except Exception as e:
            nameko_logger.error(
                f'get buildings {organization_id} {project_id} error.{str(e)}')
            return -1, f'get buildings {organization_id} {project_id} failed: {str(e)}'

    @rpc
    def edit_building(self, building_id, buildingInfo):
        try:
            with sqlSession.sqliteSession().getSession() as session:
                building = session.query(Building).filter(
                    Building.building_id == building_id).first()
                building.building_alias = buildingInfo['building_alias']
                building.building_desc_file = buildingInfo['building_desc_file']
                building.building_model_file = buildingInfo['building_model_file']
                session.commit()
                return 0, f'edit building {building_id} successfully'

        except Exception as e:
            nameko_logger.error(f'edit building {building_id} error.{str(e)}')
            return -1, f'edit building {building_id} failed: {str(e)}'
