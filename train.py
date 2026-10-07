import os
import pickle
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from datasets import getds
from word2sequence import Word2seqence
from models import ClassficicationModel

EPOCHS = 20
LR = 1e-5
BEST_PATH = "./model_output/best.pt"


def main():
    os.makedirs("./model_output", exist_ok=True)

    ws = Word2seqence()
    ws = pickle.load(open("./smsspam.data", mode="rb"))
    train_loader, test_loader = getds(ws)

    criterion = nn.CrossEntropyLoss()
    model = ClassficicationModel()
    optimizer = Adam(params=model.parameters(), lr=LR)

    best_acc = 0.0
    losses, accuracies = [], []

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x = torch.tensor(x, dtype=torch.long)   # (batch_size, seq_len)
            y = torch.tensor(y, dtype=torch.long)   # (batch_size,)
            logit = model(x)
            loss = criterion(logit, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss)

        # 每 4 轮在测试集上评估一次，并保存当前最佳模型
        if (epoch + 1) % 4 == 0:
            acc = evaluate(model, test_loader)
            accuracies.append(acc)
            tqdm.write(f"epoch {epoch + 1}: loss={epoch_loss:.4f}  test_acc={acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), BEST_PATH)

    print("losses     :", [round(l, 3) for l in losses])
    print("accuracies :", [round(a, 4) for a in accuracies])
    print(f"best test accuracy: {best_acc:.4f}  (saved to {BEST_PATH})")


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        pred = torch.argmax(model(x), dim=-1)
        correct += torch.eq(pred, y).sum().item()
        total += len(y)
    return correct / total


if __name__ == "__main__":
    main()
