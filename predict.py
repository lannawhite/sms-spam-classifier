import pickle
import jieba
import torch
import sys

from word2sequence import Word2seqence
from models import ClassficicationModel

SEQ_LEN = 50
WEIGHTS = "./model_output/best.pt"


def load():
    ws = Word2seqence()
    ws = pickle.load(open("./smsspam.data", "rb"))
    model = ClassficicationModel()
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.eval()
    return ws, model


def predict(text, ws, model):
    # 标签映射见 datasets.py：1=ham, 0=spam → 输出 logit 的 dim0=spam, dim1=ham
    ids = ws.transform(jieba.lcut(text.lower()), SEQ_LEN)
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=-1)[0]
    spam_p = probs[0].item()
    ham_p = probs[1].item()
    label = "spam" if spam_p > ham_p else "ham"
    return label, round(spam_p, 4)


if __name__ == "__main__":
    ws, model = load()
    samples = [
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May",
        "Ok lar... Joking wif u oni...",
    ]
    if len(sys.argv) > 1:
        samples = [" ".join(sys.argv[1:])]
    for s in samples:
        label, p = predict(s, ws, model)
        print(f"[{label}] (spam={p})  {s}")
