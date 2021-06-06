import torch
import torch.nn as nn
import torch.nn.functional as F
import neighbor_feature



class Pct_seg(nn.Module):
    def __init__(self):
        super(Pct_seg, self).__init__()

        self.conv1 = nn.Conv1d(628, 512, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(512, 256, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(256,128 , kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(512,256, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3=nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        self.pt_last = Point_Transformer_Last(channels=256)

        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 512, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(512),
                                        nn.LeakyReLU(negative_slope=0.2),
                                        nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(256),
                                       nn.LeakyReLU(negative_slope=0.2)

                                       )




    def forward(self, x,index):
#--------------------- Input Embedding --------------------
# Neighbor Embedding: LBR --> LBR --> LBR --> SG
        #x:(B,N,628)
        x = x.permute(0, 2, 1)
        # x (B,628,N)
        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.permute(0, 2, 1)
        # (B,N,128)
        # 连接邻接特征
        x=neighbor_feature.cat_neighbor_features(x,index)
        x=torch.unsqueeze(x,dim=0)
        #(1,n,128*4=512)
        x = x.permute(0, 2, 1)
        #(1,512,n)
        x = F.relu(self.bn4(self.conv4(x)))
        #(1,256,n)

        self.before_transformer=x
#----------------- Self Attention -------------------------------------


        #(1,256,N)
        #print(x.shape)
        x = self.pt_last(x)

        #x = x.permute(0, 2, 1)
        #x=x+self.before_transformer
        #(1,256, N)
        x = torch.cat([x, self.before_transformer], dim=1) # 256+256=512
        self.face_feature = self.conv_fuse(x)                # in:512, out:256

# Point Feature --> Global Feature --> LBRD --> LBR --> Linear --> Predict label 

        return self.face_feature

class Point_Transformer_Last(nn.Module):
    def __init__(self, channels=256):
        super(Point_Transformer_Last, self).__init__()

        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        #self.sa2 = SA_Layer(channels)
        #self.sa3 = SA_Layer(channels)
        #self.sa4 = SA_Layer(channels)

    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()

        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        #x2 = self.sa2(x1)
        #x3 = self.sa3(x2)
        #x4 = self.sa4(x3)
        #x = torch.cat([x1,x], dim=1)

        return x1

# self attention layer
class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x