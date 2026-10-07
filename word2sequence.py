import jieba


class Word2seqence:
    def __init__(self):
        self.UNK=0
        self.PAD=1
        self.word2id={"UNK":self.UNK,"PAD":self.PAD}
        self.id2word={self.UNK:"UNK",self.PAD:"PAD"}
        self.counter={}



    def fit(self,sentence):#准备好可能有用的词，一行行处理目的是为了提高效率，降低资源消耗
        """
        统计句子sentence的单词出现的次数，保存到self.counter中
        :param sentence: [word1,word2,...]
        :return: 无
        """
        for word in sentence:
            self.counter[word]=self.counter.get(word,0)+1
    def build_vob(self,min_count=2,max_count=None,max_features=None):
        """
        过滤少于min_count或者多于max_count的词，建立max_features词的词典
        :param min_count:
        :param max_count:
        :param max_features:
        :return:
        """
        self.counter=dict([item for item in self.counter.items()
                           if item[1]>=min_count])
        if max_count:
            self.counter = dict([item for item in self.counter.items()
                                 if item[1] <= max_count])
        temp=sorted([item for item in self.counter.items()],key=lambda x:x[1],
                    reverse=True)[:max_features]
        for item in temp:
            id=len(self.word2id)
            self.word2id[item[0]]=id
            self.id2word[id]=item[0]
    def transform(self,sentence,sentence_len=10):
        """
        把句子的词的列表转成数字列表 [word1,word2,...]-->[1,3,....]
        :param sentence: [word1,word2,...]
        :return: 数字的列表[1,3,....]
        """
        indices=[self.word2id.get(word,self.UNK) for word in sentence]
        if len(indices)<sentence_len: #如果句子的长度不足sentence_len，则需要补足
            indices+=[self.PAD]*(sentence_len-len(indices))
        return indices[:sentence_len]

# print(__name__)
if __name__ == '__main__':
    import pickle
    print("运行main的内容")
    ws=Word2seqence()
    file=open("SMSSpamCollection",mode="rt",encoding="utf8")
    while True:
        line=file.readline()
        if line!='' and line!='\n':
            ws.fit(jieba.lcut(line.split("\t")[1].strip().lower()))
        if not line:
            break
    ws.build_vob(0,max_features=100000)
    pickle.dump(ws,open("smsspam.data","wb"))#保存字典对象，方便其它模块调用
    # ws.fit("Dont worry. I guess he's busy.".lower().split())
    # ws.fit("worry he's .".lower().split())
    # ws.fit("worry guess he's .".lower().split())
    # ws.fit("worry guess .".lower().split())
    # ws.fit("worry ".lower().split())
    # ws.build_vob(2,max_features=100)
    # print(ws.transform("worry guess hello".split(),20))
    # print()