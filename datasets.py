import torch
from torch.utils.data import Dataset,DataLoader,random_split

from word2sequence import Word2seqence
import pickle
import jieba

class SmssDataset(Dataset):
    """
    Dataset中保存的数据应该据有list列表的特征,nlp中一般都是list类型
    """
    def __init__(self,path="SMSSpamCollection",ws:Word2seqence=None):
        self.xs=[]
        self.ys=[]
        file = open(path, mode="rt", encoding="utf8")
        for line in file:
            label,sentence=line.strip().split("\t")
            self.ys.append(1 if label=='ham' else 0)
            self.xs.append(ws.transform(jieba.lcut(sentence),50))
    def __getitem__(self, index):
        return self.xs[index],self.ys[index] #(x,y)返回元组数据
    def __len__(self):
        """
        :return: 返回数据的总数据量
        """
        return len(self.ys)



def collate_fn(batch):
    return list(zip(*batch))
def getds(ws:Word2seqence):
    ds = SmssDataset(ws=ws)
    lenOfTrain = int(len(ds) * 0.9)
    lenOfTest = len(ds) - lenOfTrain
    train_ds, test_ds = random_split(dataset=ds, lengths=[lenOfTrain, lenOfTest])
    train_loader = DataLoader(dataset=train_ds, batch_size=16, collate_fn=collate_fn, shuffle=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=16, collate_fn=collate_fn, shuffle=False)
    return train_loader,test_loader

if __name__ == '__main__':
    ds=SmssDataset()  #构造对象其实就调用__init__方法
    # print(len(ds))  #调用了类中定义的__len__方法
    # 调用类中定义的__getitem__方法，[]中的数字就是数量的索引值，
    # 它会传给方法__getitem__方法中的index参数
    # print(ds[1],ds[2],ds[1000])
    # print(ds[0:10])
    # print(ds[0])
    # print(ds[1])
    lenOfTrain=int(len(ds)*0.9)
    lenOfTest=len(ds)-lenOfTrain
    train_ds,test_ds=random_split(dataset=ds,lengths=[lenOfTrain,lenOfTest]) #根据lengths参数的个数返回数据集列表
    loader=DataLoader(dataset=train_ds,batch_size=16,collate_fn=collate_fn,shuffle=False)
    for x,y in train_ds:
        # print(list(x))
        x=torch.tensor(data=x,dtype=torch.long) #(batch_size,seq_len)->(16,50)
        # print(list(y))
        y=torch.tensor(data=y,dtype=torch.int) #(batch_size)
        print(x)
        print(y)
        break
