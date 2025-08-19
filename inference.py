import torch
from model import make_model


def decode(model, src, max_len, start_symbol):
    """
    推理
    """
    memory = model.encode(src)
    # 解码器输入以开始符号(BOS)开始
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len - 1):

        next_word = model.predict_next_word(src, ys)

        # 将预测的词拼接到当前序列中
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)

        # 如果预测的是EOS符号 (这里我们假设也用1), 则停止
        if next_word == 1 and i > 0:
            break

    return ys


def run_inference():
    vocab_size = 11
    SEQ_LEN = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = make_model(src_vocab=vocab_size, tgt_vocab=vocab_size, hidden_size=64, num_layers=2, num_heads=4, ffn=256,
                       device=device, dropout=0.1)

    model.load_state_dict(torch.load('transformer_copy_model.pth', map_location=device))
    model.to(device)
    model.eval()

    # 创建一个测试序列
    src = torch.randint(1, vocab_size, size=(1, SEQ_LEN), device=device)

    print(f"Model input:   {src}")

    output = decode(model, src, max_len=SEQ_LEN, start_symbol=1)

    print(f"Model output:   {output}")


if __name__ == "__main__":
    run_inference()