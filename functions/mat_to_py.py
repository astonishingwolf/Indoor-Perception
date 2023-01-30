import numpy as np
import h5py
import numpy as np
import glob
from scipy.io import loadmat


class PCNIData():
    def __init__(self, data_path, list_concat = True, eg = 0) -> None:
         self.data_path = data_path
         self.list_concat = list_concat
         self.eg = eg
    
    def mat_data(self,datatype):
        data = []
        if datatype == 'image':
            filenames = glob.glob(self.data_path + '\\imgdata\\*')
        else:
            filenames = glob.glob(self.data_path + '\\pcdata\\*')
        
        if self.eg:
            s = filenames[0].split('_')
            s[-2] = str(self.eg)
            filenames = ['_'.join(s)]
            print('loading ',filenames[0])
            
        for file in filenames:
            f = h5py.File(file,'r')
            data.append(f)
        return data
    
    @staticmethod
    def list_data(data,datatype):
        if datatype == 'image':
            pc_list = []
            for j in range(len(data)):
                for i in range(data[j]['pt'].shape[1]):
                    ref = data[j]['pt'][0][i]
                    pc_list.append(np.array(data[j][ref]).transpose(2,1,0))
        else:
            pc_list = []
            for j in range(len(data)):
                for i in range(data[j]['pt'].shape[1]):
                    ref = data[j]['pt'][0][i]
                    pc_list.append(np.array(data[j][ref]).transpose())
        
        return pc_list

    def __call__(self,datatype):
        data = self.mat_data(datatype)
        if self.list_concat:
            data = self.list_data(data,datatype)

        return data
            

