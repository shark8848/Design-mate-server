
from nameko.rpc import rpc
from sqlalchemy import create_engine, Column, String, Integer, CHAR
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import sys
sys.path.append("..")
#from apocolib.apocolog4p import apoLogger as apolog
from apocolib import apocoIAServerConfigurationManager as iaConMg

dbUrl = iaConMg.db_url

class sqliteSession:
    @staticmethod
    def getSession():

        engine = create_engine(dbUrl)
        Session = sessionmaker(bind=engine)
#        apolog.info("get sqlite db session successfully")
        return Session()
