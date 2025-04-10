from sqlalchemy import create_engine, Column, String, Integer, CHAR
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

dbUrl = 'sqlite:////home/apoco/ai_micro_services/apoco_intelligent_inalytics/database/apoco_ai_base_data.db'

class sqliteSession:
    @staticmethod
    def getSession():

        engine = create_engine(dbUrl)
        Session = sessionmaker(bind=engine)
        return Session()
