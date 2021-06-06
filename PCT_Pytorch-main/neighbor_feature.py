import h5py
import scipy.io as scio
import numpy
import os
import torch

#index表示第几个面片(index=1,2,...)
#neighbor_graph是一个邻接向量（3*N,）
#每次输出该index的一个邻接面片序号
def near_face(index,neighbor_graph):
    for i in range(3):
        yield neighbor_graph[(index-1)*3+i]



#data (B,N,D)
#index 表明目前的data的编号(1,2,...
#dir 邻接矩阵的目录
#返回连接三个邻接面片特征的新data(B,N,4D)
def cat_neighbor_features(data,index,dir):
    neighbor_dir=os.path.join(dir,str(index)+'.mat')
    neighbor_v = scio.loadmat(neighbor_dir)['y'].reshape(-1)
    data=torch.squeeze(data,dim=0)
    N,D=data.shape
    new_feature=[]
    #i表明目前处理的面片的序号，i=(1,2,...,N)
    for i in range(1,N+1):
        ori_face=data[i-1]
        now_face=ori_face
        for face_index in near_face(i,neighbor_v):
            #print(face_index)
            
            ng_feature=data[face_index-1]
            #print(now_face.shape)
            now_face=torch.cat([now_face,ori_face-ng_feature])
        new_feature.append(now_face.numpy())
    new_data=torch.Tensor(new_feature)
    return new_data
            