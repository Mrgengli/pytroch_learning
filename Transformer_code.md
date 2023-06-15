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

![image](https://github.com/Mrgengli/pytroch_learning/blob/main/1.png?raw=true)

* 编码我们的位置编码器的一种直观方式如下所示：
```
class PositionalEncoder(nn.Module): 
    def __init__(self, d_model, max_seq_len = 80): 
        super().__init__() 
        self.d_model = d_model   
        
        # 创建常量 'pe' 矩阵，其值取决于
        # pos 和 i 
        pe = torch .zeros(max_seq_len, d_model) 
        for pos in range(max_seq_len): 
            for i in range(0, d_model, 2): 
                pe[pos, i] = \ 
                math.sin(pos / (10000 ** ((2 * i )/d_model))) 
                pe[pos, i + 1] = \ 
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model))) 
                
        pe = pe.unsqueeze(0) #将pe的0维度扩展一个维度  [max_seq_len, d_model] ->[1,max_seq_len, d_model]
        # ？？？？为什么要用这个？？？  后面切片的时候不是两个维度？？
        self .register_buffer('pe', pe)  #将pe申请一片tensor缓存，不进行梯度下降
 
    
    def forward(self, x):   #这里输入的应该是embedding后的输入
        # 使嵌入相对较大
        x = x * math.sqrt(self.d_model)   # 在这里将x中的每一个值都乘以 self.d_model 将数值扩大，避免梯度消失。
        # 添加常量到嵌入
        seq_len = x.size(1) # 获得这句话有多少个单词
        x = x + Variable(self.pe[:,:seq_len], requires_grad=False ).cuda()
        return x
  ```
### * pe = pe.unsqueeze(0)
* pe: 这个部分是一个张量(tensor)。
* .unsqueeze(0): 这个部分调用了PyTorch的unsqueeze()函数，并传递参数0。这个函数可以在给定维度上增加新的维度。在这里，它将在第0个维度上增加一个新的维度。
因此，pe = pe.unsqueeze(0) 的含义是将一个原本形状为 **(n,d)** 的张量 

pe 转换成一个形状为** (1,n,d)** 的三维张量。通过在第0个维度上添加一个大小为1的新维度。






