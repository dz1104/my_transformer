import torch

class SimpleBatch:
    """
    一个简单的批次对象，用于封装一个批次的数据和掩码。
    """
    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            # 目标输入 (decoder input) 是去掉最后一个token的目标序列
            self.tgt = tgt[:, :-1]
            # 真实目标 (ground truth) 是去掉第一个token的目标序列
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            # 计算一个批次中非padding token的总数，用于计算平均损失
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        subsequent_mask = torch.tril(torch.ones(tgt.size(-1), tgt.size(-1))).bool().to(tgt.device)
        tgt_mask = tgt_mask & subsequent_mask
        return tgt_mask


def data_gen(vocab_size, batch_size, num_batches, seq_len, device):
    """
    生成随机数据用于训练.
    """
    for i in range(num_batches):

        data = torch.randint(1, vocab_size, size=(batch_size, seq_len), device=device)

        data[:, 0] = 1

        src = data.clone()
        tgt = data.clone()

        yield SimpleBatch(src, tgt, 0)