[English](README.md) | [中文](README_zh.md)

# 短信垃圾邮件分类器（SMS Spam Classifier）

> 在短信（SMS）文本上做**二分类**（spam 垃圾 / ham 正常），**从零实现**——用自写的 word2sequence
> 词表表示，喂给一个全连接网络。没有预训练词向量，也没有 BERT。

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-EE4C2C?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 项目概览

| 项目 | 说明 |
|---|---|
| **任务** | 文本二分类——判断短信是 `spam`（垃圾）还是 `ham`（正常） |
| **方法** | 在**自构建**的 word2sequence 词表表示上接一个全连接网络 |
| **框架** | PyTorch |
| **分词** | `jieba` 分词 + 自定义词表（`UNK` 未知词 / `PAD` 填充位） |
| **数据集** | [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) —— 5,572 条英文短信，以制表符分隔 `标签<TAB>文本` |
| **开源协议** | MIT |

---

## 为什么要从零实现

直接微调 `bert-base-chinese` 报个 98% 很简单。这个项目的意义在于**端到端理解最基础的 NLP 流水线**——一句话如何变成定长向量、嵌入表（embedding table）如何从零学出来，以及一个朴素的浅层分类器能抓到什么、抓不到什么。

亲手把每个环节写出来，才能看清它的失败模式，这也是做这个项目的全部意义：

- **未登录词（OOV）** —— 词级分词的主要失败点：每个没见过的词都会塌缩成 `UNK`
- **定长截断** —— 超过 50 个 token 的短信会被切断；短于 50 的会被 padding 填充，稀释信号
- **忽略词序** —— 全连接头把每条短信看作一串定长 token id 的"词袋"，分不清 *"not spam"* 和 *"spam not"*

## 网络结构

```
原始短信
  │
  ├─ jieba.lcut(text)                  分词
  ├─ Word2Sequence                     构建词表（UNK=0, PAD=1, 低频词过滤）
  │                                     → 统一截断/填充到定长 50 个 id
  ├─ Embedding(100000, 100)            (batch,50) → (batch,50,100)
  ├─ flatten                           (batch,50,100) → (batch,5000)
  ├─ Linear(5000 → 500) + sigmoid
  ├─ Linear(500 → 2)   + sigmoid      对 [spam, ham] 输出 logits
  └─ CrossEntropyLoss，预测 = argmax
```

> `标签`映射（见 `datasets.py`）：`ham → 1`，`spam → 0`，因此输出 logit 顺序为 `[spam, ham]`。

## 目录结构

```
sms-spam-classifier/
├── SMSSpamCollection     # 数据集（UCI），5572 行，标签<TAB>文本
├── word2sequence.py      # 自定义词表构建器（fit / build_vob / transform）+ pickle 落盘
├── smsspam.data          # 预构建词表（由 word2sequence.py 生成）
├── datasets.py           # torch Dataset：加载 + 分词 + 9:1 训练/测试划分
├── models.py             # ClassficicationModel —— Embedding → 2 层 Linear（sigmoid）
├── train.py              # 训练循环（Adam，20 epoch，每 4 轮评估，保存最优）
├── predict.py            # 加载权重做单句推理
├── requirements.txt
└── README.md
```

## 快速开始

```bash
pip install -r requirements.txt

# 1. （可选）重建词表 —— smsspam.data 已经随仓库附带，通常不需要
python word2sequence.py

# 2. 训练 —— 每 4 个 epoch 打印一次 loss + 测试集准确率，保存 model_output/best.pt
python train.py

# 3. 给一条短信分类
python predict.py "Free entry in 2 a wkly comp to win FA Cup final tkts"
# → [spam] (spam=0.xxxx)  ...
```

## 效果

| 模型 | 测试准确率 |
|---|---|
| Word2Sequence + 2 层全连接（本仓库） | **TODO —— 跑完 `train.py` 后填入**（该公开数据集上预期约 97–98%） |
| TF-IDF + 逻辑回归（参考基线） | TODO |

> ⚠️ 上面的准确率**尚未实测**——请在本地跑完 `train.py` 后再填入真实数字。
> 预期区间引用自该公开数据集上的常见结果；在拿到真实数值之前，不要直接抄进简历。

## 诚实的局限性

- `jieba` 是**中文**分词器。对英文短信来说，用空格 + 正则分词会更地道——这是一处可干净替换的地方。
- 50 的定长会切断长短信、填充短短信。
- 全连接头忽略词序；部分垃圾邮件模式（否定、强调）是和顺序相关的。

## 接下来想做的

- [ ] 把 `jieba` 换成英文友好的分词（正则 / `nltk`）
- [ ] 加一个 RNN / LSTM 分支，量化序列建模到底带来多少提升
- [ ] 与 TF-IDF + 逻辑回归做基准对比，体现深度模型的增量收益
- [ ] 用 **Gradio** 一个 `gr.Interface` 调用包成可交互 demo，让招聘方点开就能试

---

<sub>由 <a href="https://github.com/lannawhite">@lannawhite</a> 构建 · 北部湾大学 人工智能专业 本科生</sub>
