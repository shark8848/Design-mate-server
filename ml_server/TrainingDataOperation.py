from DataModel import Glass,InsulationMaterial
import sqliteSession as sqlSession
import pdb
import numpy as np

class GlassEntity:
    def __init__(self, s_d,thickness, coating, hollow_material, price, k):
        self.S_D = s_d
        self.thickness = thickness
        self.coating = coating
        self.hollow_material = hollow_material
        self.price = price
        self.k = k

class InsulationMaterialEntity:
    def __init__(self, thickness, price, k):
        self.thickness = thickness
        self.price = price
        self.k = k

class GlassDataOperation:

    def __init__(self):
#        pdb.set_trace()

        self.data = []
        self.index = {}
        self.glass = None
        self.P_K = []
        print("单/双玻(1/2)  厚度  镀膜  中空材料(0~?)  单价   K")
        with sqlSession.sqliteSession().getSession() as session:
            glass = session.query(Glass).all()
            for g in glass:
                self.P_K.append([round(g.price,2),round(g.K,4)])
                print(f'g - {g.S_D} - {g.thickness} - {g.coating} - {g.hollow_material} - {g.price} - {g.K} ')
                G_E = GlassEntity(
                        g.S_D, 
                        round(g.thickness,4),
                        round(g.coating,4),
                        g.hollow_material,
                        round(g.price,2),
                        round(g.K,4)
                        )
                self.data.append(G_E)
                self.index[float(g.K)] = g

    def search_by_k(self, k):
        return self.index.get(k)

    def get_P_K(self):
        return np.array(self.P_K)
    def get_data(self):
        return np.array(self.data)

    def get_title(self,lg):
        if lg == 'Cn':
            return "单/双玻(1/2)  厚度  镀膜  中空材料(0~?)  单价   K"
        else:
            return "S_D(1/2)    thickness   coating     hollow_material     price       K"

class InsulationMaterialDataOperation:

    def __init__(self):

        self.data = []
        self.index = {}
        self.glass = None
        self.P_K = []
        print("厚度  单价   K")
        with sqlSession.sqliteSession().getSession() as session:
            insulationMaterial = session.query(InsulationMaterial).all()
            for im in insulationMaterial:
                self.P_K.append([round(im.price,2),round(im.K,4)])
                print(f'{im.thickness} - {im.price} - {im.K} ')
                IM_E = InsulationMaterialEntity(
                        round(im.thickness,4),
                        round(im.price,2),
                        round(im.K,4)
                        )
                self.data.append(IM_E)
                self.index[float(im.K)] = im

    def search_by_k(self, k):
        return self.index.get(k)

    def get_P_K(self):
        return np.array(self.P_K)
    def get_data(self):
        return np.array(self.data)

    def get_title(self,lg):
        if lg == 'Cn':
            return "厚度  单价   K"
        else:
            return "thickness   price    K"


if __name__ == '__main__':
    
    g_d = GlassDataOperation()
    # 检索 k 值为 3.57 的数据
    res = g_d.search_by_k(3.57)
    print("单/双玻(1/2)  厚度  镀膜  中空材料(0~?)  单价   K")
    print(f'g - {res.S_D} - {res.thickness} - {res.coating} - {res.hollow_material} - {res.price} - {res.K} ')
    print("单价   K")
    p_k = g_d.get_P_K()
    print(p_k)

    im_d = InsulationMaterialDataOperation()
    res = im_d.search_by_k(0.71)
    print("厚度  单价   K")
    print(f'{res.thickness} - {res.price} - {res.K} ')
    print("单价   K")
    p_k = im_d.get_P_K()
    print(p_k)
