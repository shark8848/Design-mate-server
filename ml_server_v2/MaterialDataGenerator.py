from sqlalchemy import and_
import sqliteSession as sqlSession
from DataModel import *
from sqlalchemy import PrimaryKeyConstraint

with sqlSession.sqliteSession().getSession() as session:

    # Clear the 'Glass' table before generating new data
    print("Clear the 'Glass' table before generating new data")
    session.query(Glass).delete()


    glass_data = []

    for s_d in [1, 2]:
        for base_price in session.query(GlassBasePrice).all():# 7rows
            for silver_plated in session.query(SilverPlatedPrice).filter(SilverPlatedPrice.thickness == base_price.thickness).all():#14rows
                for tempered in session.query(GlassTemperedPrice).filter(GlassTemperedPrice.thickness == base_price.thickness).all():#7rows
                    for special_process in session.query(SpecialProcessPrice).filter(SpecialProcessPrice.thickness == base_price.thickness).all():#30rows
                        if s_d == 2 and silver_plated.s_d == 2:
                            for hollow_filling in session.query(HollowFillingPrice).all():#filter(HollowFillingPrice.hollow_thickness == base_price.thickness).all():
                            #for hollow_filling in session.query(HollowFillingPrice).filter(HollowFillingPrice.hollow_thickness == base_price.thickness).all():
                                for side_strip in session.query(SideStripPrice).filter(SideStripPrice.thickness == base_price.thickness).all():
                                    
                                    b_k_value = round(1/(base_price.r + silver_plated.r + hollow_filling.r ),4)
                                    t_p_price = round(base_price.price + silver_plated.price + tempered.price + special_process.price + hollow_filling.price + side_strip.price,2)
                                    
                                    new_glass = Glass(
                                        id = len(glass_data) + 1,
                                        S_D=s_d,
                                        type='1',
                                        thickness=base_price.thickness,
                                        base_price=base_price.price,
                                        silver_plated=silver_plated.id,
                                        s_pl_price=silver_plated.price,
                                        s_pl_k_ratio=1.0,
                                        hollow_material=hollow_filling.id,
                                        b_K=b_k_value,
                                        special_process=1 if special_process.price > 0 else 0,
                                        sp_p_price=special_process.price,
                                        sp_p_k_ratio=1.0,
                                        tempered=1 if tempered.price > 0 else 0,
                                        t_price= t_p_price, #base_price.price + silver_plated.price + tempered.price + side_strip.price,
                                        t_k_ratio=1.0,
                                        t_K = b_k_value, #* s_pl_k_ratio * t_k_ratio * sp_p_k_ratio * b_K
                                        side_strip=side_strip.side_strip,
                                        side_strip_price=side_strip.price,
                                        #description=f'{s_d}层玻璃-{base_price.thickness}mm-{silver_plated.description}-{tempered.description}-{special_process.description}-{hollow_filling.description}-{side_strip.description}'
                                        description=f'{s_d}层玻璃-{silver_plated.description}-{tempered.description}-{special_process.description}-{hollow_filling.description}-{side_strip.description}'

                                    )
                                    glass_data.append(new_glass)
                        else:
                            if silver_plated.s_d == 1:
                                for side_strip in session.query(SideStripPrice).filter(SideStripPrice.thickness == base_price.thickness).all():
                                    
                                    b_k_value = round(1/(base_price.r + silver_plated.r ),4)
                                    t_p_price = round(base_price.price + silver_plated.price + tempered.price + special_process.price + side_strip.price,2)
                                    
                                    new_glass = Glass(
                                        id=len(glass_data) + 1,
                                        S_D=s_d,
                                        type='1',
                                        thickness=base_price.thickness,
                                        base_price=base_price.price,
                                        silver_plated=silver_plated.id,
                                        s_pl_price=silver_plated.price,
                                        s_pl_k_ratio=1.0,
                                        hollow_material=0,
                                        b_K=b_k_value,
                                        special_process=1 if special_process.price > 0 else 0,
                                        sp_p_price=special_process.price,
                                        sp_p_k_ratio=1.0,
                                        tempered=1 if tempered.price > 0 else 0,
                                        t_price= t_p_price,#base_price.price + silver_plated.price + tempered.price + side_strip.price,
                                        t_k_ratio=1.0,
                                        t_K = b_k_value, #* s_pl_k_ratio * t_k_ratio * sp_p_k_ratio * b_K,
                                        side_strip=side_strip.side_strip,
                                        side_strip_price=side_strip.price,
                                        #description=f'{s_d}层玻璃-{base_price.thickness}mm-{silver_plated.description}-{tempered.description}-{special_process.description}-{side_strip.description}'
                                        description=f'{s_d}层玻璃-{silver_plated.description}-{tempered.description}-{special_process.description}-None-{side_strip.description}'

                                    )
                                    glass_data.append(new_glass)

    session.add_all(glass_data)
    session.commit()
    print(f'Generated {len(glass_data)} rows of data.')

