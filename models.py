import torch
import torch.nn as nn
from torch.nn import functional as F
# class ClassficicationModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.embedding = nn.Embedding(num_embeddings=10000, embedding_dim=100)
#         self.w=nn.Parameter(data=torch.randn(100*50,2))
#         self.b=nn.Parameter(data=torch.randn(2))
#
#     def forward(self,x):
#         """
#         只接收一个参数
#         :param x: batch,seq_len
#         :return:
#         """
#         # (batch_size,seq_len)->(batch_size,seq_len,dim)->(16,50)-->(16,50,100)
#         x=self.embedding(x)
#         x=x.reshape(x.shape[0],-1)  #(16,50,100)->(16,5000)
#         out=F.sigmoid(x@self.w+self.b)   #(16,2)
#         return out

# class ClassficicationModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.embedding = nn.Embedding(num_embeddings=10000, embedding_dim=100)
#         self.linear=nn.Linear(in_features=50*100,out_features=2,bias=True)
#
#     def forward(self,x):
#         """
#         只接收一个参数
#         :param x: batch,seq_len
#         :return:
#         """
#         # (batch_size,seq_len)->(batch_size,seq_len,dim)->(16,50)-->(16,50,100)
#         x=self.embedding(x)
#         x=x.reshape(x.shape[0],-1)  #(16,50,100)->(16,5000)
#         out=F.sigmoid(self.linear(x)) #(16,2)
#         return out


class ClassficicationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=100000, embedding_dim=100)  # vocab size from smsspam.data (build_vob max_features=100000)
        self.linear1=nn.Linear(in_features=50*100,out_features=500,bias=True)
        self.linear2=nn.Linear(in_features=500,out_features=2,bias=True)
    def forward(self,x):
        """
        只接收一个参数
        :param x: batch,seq_len
        :return:
        """
        # (batch_size,seq_len)->(batch_size,seq_len,dim)->(16,50)-->(16,50,100)
        x=self.embedding(x)
        x=x.reshape(x.shape[0],-1)  #(16,50,100)->(16,5000)
        x=F.sigmoid(self.linear1(x)) #(16,5000)->(16,500)
        out=F.sigmoid(self.linear2(x)) #(16,500)->(16,2)
        return out