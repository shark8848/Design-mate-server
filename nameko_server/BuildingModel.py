from sqlalchemy import create_engine, Column, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()

class Building(Base):
    __tablename__ = 'buildings'

    building_id = Column(String(20), primary_key=True, unique=True)
    organization_id = Column(String(20), nullable=False)
    project_id = Column(String(20), nullable=False)
    building_alias = Column(String(200), nullable=False)
    building_desc_file = Column(String(200))
    building_model_file = Column(String(200))
    create_time = Column(DateTime, nullable=False)

    # Define relationships if you have 'organizations' and 'projects' tables
    #organization = relationship("Organization")
    #project = relationship("Project")