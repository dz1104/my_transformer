import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from model import make_model
from data import data_gen, SimpleBatch
# from utils import create_masks


class LabelSmoothing(nn.Module):
    "实现标签平滑"

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


def run_epoch(data_iter, model, loss_compute, optimizer, device):
    "标准的训练和日志记录函数"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        src = batch.src.to(device)
        tgt = batch.tgt.to(device)
        tgt_y = batch.tgt_y.to(device)
        # src_mask, tgt_mask = create_masks(src, tgt)

        out = model(src, tgt)

        loss = loss_compute(out.contiguous().view(-1, out.size(-1)),
                            tgt_y.contiguous().view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss.item() / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0

    return total_loss / total_tokens


# --- 主训练函数 ---
def main():
    # 词汇表大小 (0 for pad, 1 for BOS/EOS, 2-10 for numbers)
    vocab_size = 11
    # 序列长度
    SEQ_LEN = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 模型初始化
    model = make_model(src_vocab=vocab_size,tgt_vocab=vocab_size,hidden_size=64, num_layers=2, num_heads=4, ffn=256, device=device, dropout=0.1)
    model.to(device)

    # 2. 损失函数
    criterion = LabelSmoothing(size=vocab_size, padding_idx=0, smoothing=0.1)

    # 3. 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

    # 4. 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        print(f"--- Epoch {epoch + 1} ---")
        # 每个epoch生成新的数据
        data_iterator = data_gen(vocab_size=vocab_size, batch_size=32, num_batches=100, seq_len=SEQ_LEN, device=device)
        train_loss = run_epoch(data_iterator, model, criterion, optimizer, device)
        print(f"Epoch {epoch + 1} Training Loss: {train_loss}")

    # 保存模型
    torch.save(model.state_dict(), 'transformer_copy_model.pth')
    print("Model saved.")


if __name__ == "__main__":
    main()