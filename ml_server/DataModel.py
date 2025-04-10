from sqlalchemy import (create_engine, Column, Integer, DateTime, Text, String, Numeric, Float,ForeignKey, PrimaryKeyConstraint)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import pytz

Base = declarative_base()

class Glass(Base):
    __tablename__ = 'glass'
    #__table_args__ = (PrimaryKeyConstraint('base_price', 'b_K'),
    id = Column(Integer, primary_key=True)
    type = Column(String, nullable=False)
    S_D = Column(Integer, nullable=False, default=1)
    thickness = Column(Float, nullable=False, default=0)
    base_price = Column(Float, nullable=False, default=0.0)
    silver_plated = Column(Integer, nullable=False, default=0)
    s_pl_price = Column(Float, nullable=False, default=0.0)
    s_pl_k_ratio = Column(Float, nullable=False, default=1.0)
    hollow_material = Column(Float, nullable=False, default=0)
    b_K = Column(Float,nullable=False, default=0.0)
    special_process = Column(Integer, nullable=False, default=0)
    sp_p_price = Column(Float, nullable=False, default=0.0)
    sp_p_k_ratio = Column(Float, nullable=False, default=1.0)
    tempered = Column(Integer, nullable=False, default=0)
    t_k_ratio = Column(Float, nullable=False, default=1.0)
    side_strip = Column(Integer, nullable=False, default=0)
    t_price = Column(Float, nullable=False, default=0.0)
    t_K = Column(Float, nullable=False, default=0.0)
    description = Column(String, nullable=False)
    side_strip_price = Column(Float, nullable=False, default=0.0)


class GlassBasePrice(Base):
    __tablename__ = 'glass_base_price'

    id = Column(Integer, primary_key=True)
    thickness = Column(Numeric, nullable=False)
    price = Column(Numeric(10, 2), nullable=False)
    description = Column(String, nullable=False)
    r = Column(Numeric, nullable=False, default=0.0)


class GlassTemperedPrice(Base):
    __tablename__ = 'glass_tempered_price'

    id = Column(Integer, primary_key=True)
    description = Column(String, nullable=False)
    thickness = Column(Integer, nullable=False)
    price = Column(Numeric, nullable=False)


class GlassThicknessSpecifications(Base):
    __tablename__ = 'glass_thickness_specifications'

    id = Column(Integer, primary_key=True)
    thickness = Column(Numeric, nullable=False)


class HollowFillingMaterial(Base):
    __tablename__ = 'hollow_filling_material'

    id = Column(Integer, primary_key=True)
    description = Column(String)


class HollowFillingPrice(Base):
    __tablename__ = 'hollow_filling_price'

    id = Column(Integer, primary_key=True)
    hollow_filling = Column(Integer)
    description = Column(String)
    hollow_thickness = Column(Numeric, nullable=False, default=0.0)
    price = Column(Numeric, nullable=False, default=0.0)
    r = Column(Numeric, nullable=False, default=0.0)


class SideStripPrice(Base):
    __tablename__ = 'side_strip_price'

    id = Column(Integer, primary_key=True)
    thickness = Column(Integer, nullable=False)
    side_strip = Column(Integer, ForeignKey('side_strip_type.id'), nullable=False)
    description = Column(String, nullable=False)
    price = Column(Numeric(10, 2), nullable=False)


class SideStripType(Base):
    __tablename__ = 'side_strip_type'
    id = Column(Integer, primary_key=True)
    description = Column(String, nullable=False)
    side_strip_prices = relationship("SideStripPrice", backref="side_strip_type")

class SilverPlatedPrice(Base):
    __tablename__ = 'silver_plated_price'
    id = Column(Integer, primary_key=True)
    description = Column(String)
    thickness = Column(Integer)
    s_d = Column(Integer, default=1)
    price = Column(Numeric)
    r = Column(Numeric)

class SpecialProcessPrice(Base):
    __tablename__ = 'special_process_price'
    id = Column(Integer, primary_key=True)
    description = Column(String, nullable=False)
    thickness = Column(Numeric, nullable=False)
    price = Column(Numeric(10, 2), nullable=False)

class TaskHistory(Base):
    __tablename__ = 'task_history'

    task_id = Column(Integer, primary_key=True, unique=True, nullable=False)
    queue_name = Column(String, nullable=False)
    file_name = Column(String, nullable=False)
    out_file = Column(String, nullable=False)
    status = Column(String, nullable=False)
    #created_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=lambda: datetime.now(pytz.timezone('Asia/Shanghai')))
    completed_at = Column(DateTime)


#外墙
class ImParameters(Base):
    __tablename__ = 'im_parameters'
    id = Column(Integer,primary_key=True,unique=True,nullable=False)
    type_number = Column(String,nullable=False)
    level = Column(Integer,nullable=False)
    description = Column(String,nullable=False)
    thickness = Column(Float,nullable=False)
    price = Column(Float,nullable=False)
    K = Column(Float,nullable=False)

class WfMaterials(Base):
    __tablename__='wf_materials'
    id=Column(Integer,primary_key=True,unique=True,nullable=False)
    wf_material=Column(String,nullable=False)
    profile_section=Column(Float,nullable=False)
    wf_type=Column(String,nullable=False)
    window_area=Column(Float,nullable=False)
    wf_area=Column(Float,nullable=False)
    window_frame_area_ratio=Column(Float,nullable=False)
    K=Column(Float,nullable=False)
    price=Column(Float,nullable=False)
    description=Column(String,nullable=False)

