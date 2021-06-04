from torch.utils.data import Dataset
import torch
import numpy as np

'''


1.每次输出一个模型的面片、label,输出形式是np.array
面片形状(N,628),label形状(N,)

2.如何区分不同的类别
在调用时传入参数index指明当前训练/测试的类别
参数partition指明是训练/测试

3.load_data_and_label函数
函数接受两个目录，分别指明训练集和标签，同时需要指明模式和类别索引
返回两个列表，列表中的元素均为np.array对象

4.还没有处理260~279的无效输入

'''


def load_data_and_label(data_dir,label_dir,partition,index):
    start,cnt=index*20,12
    if partition=='test':
        cnt=8
        start+=12
    data_list,label_list=[],[]
    for i in range(1,cnt+1):
        DATA_DIR=os.path.join(data_dir,str(start+i)+'.off.txt')
        LABEL_DIR=os.path.join(label_dir,str(i+start)+'.seg')
        data=np.loadtxt(DATA_DIR)
        label=np.loadtxt(LABEL_DIR)
        data_list.append(data)
        label_list.append(label)
    return data_list,label_list

class PsbDataset(Dataset):
    def __init__(self,txt_path,partition='train',index):
        self.data,self.label=load_data_and_label(data_dir,partition,index)
        
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self,idx):
        data=self.data[idx]
        label=self.label[idx]
        return data,label
        