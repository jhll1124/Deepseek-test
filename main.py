from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pyarrow.parquet as pq
import jieba
import torch
import torch.nn as nn
import os
import zhconv


# class definition
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.fc = nn.Linear(embedding_dim, vocab_size)
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x)
        return x

class TextDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        input_seq, target_seq = self.samples[idx]
        return torch.LongTensor(input_seq), torch.LongTensor(target_seq)


def print_model_info(model):
    """打印模型关键信息"""
    print("="*45 + " 模型信息 " + "="*45)
    print(f"▪ 模型名称: {model.__class__.__name__}")
    print(f"▪ 设备位置: {next(model.parameters()).device}")
    print(f"▪ 训练模式: {model.training}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"▪ 总参数量: {total_params:,}")
    print(f"▪ 可训练参数量: {trainable_params:,}")
    print(f"▪ 冻结参数量: {total_params - trainable_params:,}")
    for name, layer in model.named_children():
        num_params = sum(p.numel() for p in layer.parameters())
        print(f"[{name.upper()}]")
        print(f"  └─ 类型: {layer.__class__.__name__}")
        print(f"  └─ 参数量: {num_params:,}")
        if hasattr(layer, 'out_features'):  # 适用于线性层等
            print(f"  └─ 输出维度: {layer.out_features}")
        elif hasattr(layer, 'num_layers'):  # 适用于Transformer堆叠层
            print(f"  └─ 层数: {layer.num_layers}")
    if hasattr(model, 'config'):
        config = model.config
        print(f"词表大小: {getattr(config, 'vocab_size', 'N/A')}")
        print(f"嵌入维度: {getattr(config, 'embedding_dim', 'N/A')}")
        print(f"隐藏层维度: {getattr(config, 'hidden_dim', 'N/A')}")
        print(f"注意力头数: {getattr(config, 'nhead', 'N/A')}")
    # 设备检查
    devices = {p.device for p in model.parameters()}
    if len(devices) > 1:
        print("\n⚠️ 警告: 模型参数分布在多个设备上!\n")
        for i, device in enumerate(devices):
            print(f"设备 {i+1}: {device}")
    else:
        print("\n✓ 所有参数位于同一设备\n")


def main():
    #! 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    #! 初始化 transformer 模型
    model = TransformerModel(
        vocab_size=10000,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=6
    ).to(device)

    #! 检查是否存在已保存模型
    if not os.path.exists("model.pth"):

        #! 读取 parquet 文件
        parquet_file = pq.ParquetFile('D:\\AI\\train-00000-of-00006.parquet')
        data = parquet_file.read().to_pandas() # .head(100)
        # data = parquet_file.read().to_pandas().head(100)
        text_column = 'text'
        tokens = []
        for text in data[text_column]:
            tokens.extend(jieba.cut(zhconv.convert(text, 'zh-cn')))
        print("总 token 数量:", len(tokens))

        #! word embedding
        # 1. 构建词表（Vocabulary）
        word2idx = {"<PAD>": 0, "<UNK>": 1, "<EOS>": 2} # 特殊符号
        idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<EOS>"} # <EOS> End Of Sequence
        # 遍历数据集统计词频
        counter = Counter()
        for token in tokens:
            counter[token] += 1
        # 选择前N个高频词构建词表（示例取前10000个词）
        vocab_size = 10000
        for word, _ in counter.most_common(vocab_size - len(word2idx)):  # 保留特殊符号位置
            if word not in word2idx:
                idx = len(word2idx)
                word2idx[word] = idx
                idx2word[idx] = word
        print(f"词表大小: {len(word2idx)}")
        # 2. 将分词转换为索引（Index）
        token_indices = [word2idx.get(t, word2idx["<UNK>"]) for t in tokens]
        # 3. 定义嵌入层（Embedding Layer）
        embedding_dim = 128  # 嵌入维度
        embedding = nn.Embedding(
            num_embeddings=vocab_size,  # 词表大小
            embedding_dim=embedding_dim,
            padding_idx=0  # 填充符索引
        ).to(device)  # 将embedding层移到GPU
        # 4. 将索引转换为嵌入向量
        input_tensor = torch.tensor([token_indices], device=device)
        embeddings = embedding(input_tensor)
        # 5. debug
        print('embeddings.shape:', embeddings.shape)
        print('embeddings.is_cuda:', embeddings.is_cuda)  # 验证是否在GPU上

        #! 构建训练数据批次
        # 将连续的token_indices分割为输入-目标对
        samples = []
        for i in range(0, len(token_indices) - seq_length, seq_length):
            input_seq = token_indices[i:i+seq_length]
            target_seq = token_indices[i+1:i+1+seq_length]  # 目标为输入右移1位
            samples.append( (input_seq, target_seq) )
        dataset = TextDataset(samples)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        #! train model
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        # 训练循环
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            for batch_inputs, batch_targets in dataloader:
                # 移动数据到CUDA
                batch_inputs = batch_inputs.to(device)   # [batch_size, seq_length]
                batch_targets = batch_targets.to(device) # [batch_size, seq_length]
                optimizer.zero_grad()
                # 前向传播
                outputs = model(batch_inputs)  # [batch_size, seq_length, vocab_size]
                # 计算损失（需要展平序列和词汇维度）
                loss = criterion(
                    outputs.view(-1, vocab_size),  # [batch*seq_length, vocab_size]
                    batch_targets.view(-1)         # [batch*seq_length]
                )
                # 反向传播与优化
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
                optimizer.step()
                total_loss += loss.item()
            # 打印损失
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")
        #! save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'word2idx': word2idx,  # 保存词表
            'idx2word': idx2word
        }, "model.pth")
        print('Model saved successfully.')
        return 0

    #! 加载模型
    print("检测到已有模型文件，加载模型中...")
    checkpoint = torch.load("model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    word2idx = checkpoint['word2idx']
    idx2word = checkpoint['idx2word']
    print_model_info(model)
    model.eval() # 切换到评估模式（关闭dropout等）
    jieba.default_logger.handlers = [] # 关闭分词日志

    #! def 推理
    def preprocess(text):
        tokens = jieba.cut(text)
        return [word2idx.get(t, word2idx["<UNK>"]) for t in tokens]
    def generate(
        model, 
        prompt, 
        max_length=500, 
        temperature=0.7, 
        top_k=50
    ):
        # 预处理输入
        input_ids = preprocess(prompt)
        generated = input_ids.copy()
        with torch.no_grad():
            for _ in range(max_length):
                # 构建输入张量
                inputs = torch.LongTensor([input_ids]).to(device)  # [1, seq_len]
                # 模型预测
                outputs = model(inputs)  # [1, seq_len, vocab_size]
                next_token_logits = outputs[0, -1, :]  # 取最后一个位置的logits
                # 概率处理
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                if top_k > 0:
                    top_values = torch.topk(next_token_logits, top_k)
                    next_token_logits[next_token_logits < top_values.values[-1]] = -float('inf')
                # 采样
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                # 实时输出
                token_str = idx2word.get(next_token, "<UNK>")
                print(token_str, end='', flush=True)  # 关键点：逐词输出
                generated.append(next_token)
                input_ids = [next_token]  # 单步生成模式
                if next_token == word2idx["<EOS>"]:  # 如果是结束符则停止生成
                    break
        # 转换回文本
        return ''.join([idx2word.get(idx, "<UNK>") for idx in generated])

    #! 推理
    while True:
        try:
            ipt = input(">>> ").strip()
            generate(model, ipt, max_length=20)
            print()
        except KeyboardInterrupt:
            print('exit')
            break


if __name__ == '__main__':
    batch_size = 1    # 根据显存调整，若CUDA内存不足则减小
    seq_length = 256    # 输入序列长度
    num_epochs = 20    # 训练轮次
    main()
    