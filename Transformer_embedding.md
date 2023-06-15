## 词嵌入
```
class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)
```
参数解释:

* vocab_size 是整数，表示词汇表的大小，即不同词汇的数量。它指定了要创建的嵌入层的**输入维度**，也就是输入的整数索引的范围。
* d_model 是整数，表示嵌入向量的维度。它指定了要创建的嵌入层的**输出维度**，也就是每个整数索引将映射到的向量的长度。

## positional encoding
