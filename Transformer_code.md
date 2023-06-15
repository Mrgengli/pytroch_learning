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

## positional encoding(位置编码)

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
        # 我们在加法之前增加嵌入值的原因是为了使位置编码相对较小。这意味着当我们将它们加在一起时，嵌入向量中的原始含义不会丢失。
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

## MASK掩码
掩蔽在变压器中起着重要作用。它有两个目的：

* 在编码器和解码器中：在输入句子中只有填充的地方将注意力输出归零。
* 在解码器中：防止解码器在预测下一个单词时在翻译句子的其余部分提前“达到峰值”。

### 为输入创建掩码很简单：
```
batch = next(iter(train_iter))   #这里的 train_iter 相当于一个dataloader的对象
# 这里假设 train_iter 是一个数据加载器或迭代器，使用 iter(train_iter) 创建一个迭代器，
并通过 next() 函数从迭代器中获取下一个批次的数据

input_seq = batch.English.transpose(0,1) 
# ？？？？？？？ 这里的English 是什么东西
# transpose(0, 1)，表示将张量的第 0 维度和第 1 维度进行交换。
# 这可能是为了符合特定模型对输入数据维度的要求。
# 一般来说，输入数据的维度是 [sequence_length, batch_size]，而 transpose(0,1) 将其转置为 [batch_size, sequence_length] 的形状。
# 这样做是为了适应一些模型对输入维度的要求，例如 Transformer 模型通常接受 [batch_size, sequence_length] 的输入。

input_pad = EN_TEXT.vocab.stoi['<pad>']
# 这行代码获取了一个特殊标记 <pad> 在英文文本词汇表 EN_TEXT.vocab 中的索引值。EN_TEXT.vocab 是一个用于存储文本数据的词汇表对象，
# stoi 表示从单词到索引的映射。通过 ['<pad>'] 访问 <pad> 这个特殊标记，并获取其在词汇表中的索引值。

# 在输入中有填充的地方创建带有 0 的掩码
input_msk = (input_seq != input_pad).unsqueeze(1)
```
* input_seq 是一个表示输入序列的张量，其形状为 [sequence_length, batch_size]。该张量中的**每个元素代表一个单词的索引或编码**。

* input_pad 是一个表示填充标记 <pad> 在词汇表中的索引值。该值通常用于标识序列的填充位置。

* (input_seq != input_pad) 会产生一个布尔张量，其形状与 input_seq 相同，每个元素表示对应位置上的单词是否等于填充标记 <pad>。如果某个位置的单词不是 <pad>，则对应位置上的值为 True，否则为 False。

* .unsqueeze(1) 将上一步得到的布尔张量的维度扩展，在第 1 维度上增加一个维度。这样操作后，布尔张量的形状变为 [sequence_length, 1, batch_size]。这个额外的维度通常用于与其他张量进行广播操作。
### 对于 target_seq，我们做同样的事情，但随后创建一个额外的步骤：
```
target_seq = batch.French.transpose(0,1)
target_pad = FR_TEXT.vocab.stoi['<pad>']
target_msk = (target_seq != target_pad).unsqueeze(1)
size = target_seq.size(1) # get seq_len for matrix
nopeak_mask = np.triu(np.ones(1, size, size),
k=1).astype('uint8')
nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0)
target_msk = target_msk & nopeak_mask
```
* target_seq = batch.French.transpose(0, 1)：获取批次中的法语序列数据，并通过 transpose(0, 1) 进行转置操作，将维度从 [sequence_length, batch_size] 转换为 [batch_size, sequence_length] 的形状。这样做是为了适应模型对输入数据维度的要求。

* target_pad = FR_TEXT.vocab.stoi['<pad>']：获取特殊标记 <pad> 在法语词汇表 FR_TEXT.vocab 中的索引值。

* (target_seq != target_pad)：产生一个布尔张量，其形状与 target_seq 相同，用于判断每个位置上的法语单词是否等于填充标记 <pad>。如果某个位置的单词不是 <pad>，则对应位置上的值为 True，否则为 False。

* .unsqueeze(1)：在第 1 维度上增加一个维度，将布尔张量的形状从 [sequence_length, batch_size] 变为 [sequence_length, 1, batch_size]。

* np.triu(np.ones(1, size, size), k=1).astype('uint8')：使用 NumPy 创建一个上三角矩阵，矩阵的形状为 (1, size, size)，其中 size 是目标序列的长度。该矩阵的上三角部分元素为 1，下三角部分和对角线元素为 0。

* nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0)：将上述创建的上三角矩阵转换为 PyTorch 张量，并将其中的非零元素设为 False，零元素设为 True。这样得到的 nopeak_mask 张量与目标序列长度相关。

* target_msk = target_msk & nopeak_mask：对目标掩码张量 target_msk 和 nopeak_mask 进行逐元素的逻辑与操作。这个操作的目的是将目标掩码中对应位置的元素，与 nopeak_mask 对应位置的元素进行逻辑与操作，以实现去除未来信息的效果。即，在目标掩码中，将未来位置的元素标记为 False，过滤掉模型不应该看到的未来信息。


