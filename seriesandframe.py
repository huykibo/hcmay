import pandas as pd  
import numpy as np
data= {
        'tuoi' : [13,14,15,16],
        'ten': ['huy','kiet','tuan',np.nan],
        'lop': ["12b3","13h4","13lk3",np.nan]
}
x= pd.DataFrame(data,index= ['a','b','c','d'])
x['thimon']=pd.Series(['anh','toan','ly',np.nan]) #them cột
# # del x["thimon"]   xóa cột bằng dell
# x.pop('thimon') xoa cotjt bằng pop
sx =x.sort_values (['ten', 'lop'] , inplace= True, ascending=[False, False]  ) # sSắp xếp true : tăng , false : giảm
print(x)