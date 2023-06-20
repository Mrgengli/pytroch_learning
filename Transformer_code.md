[transformer详细解释！](http://jalammar.github.io/illustrated-transformer/)
[哈佛大学逐行解释transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
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
* ### pe = pe.unsqueeze(0)
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
nopeak_mask = np.triu(np.ones(1, size, size),k=1).astype('uint8')
    
nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0)
 # 这个上三角的掩码 在一起   
target_msk = target_msk & nopeak_mask  # ???????这里的与操作究竟是怎么回事
# 解释：   
# result = a & b
# 在此操作中，对于每个位置 i，result[i] 的值为 a[i] and b[i]。
```
* target_seq = batch.French.transpose(0, 1)：获取批次中的法语序列数据，并通过 transpose(0, 1) 进行转置操作，将维度从 [sequence_length, batch_size] 转换为 [batch_size, sequence_length] 的形状。这样做是为了适应模型对输入数据维度的要求。

* target_pad = FR_TEXT.vocab.stoi['<pad>']：获取特殊标记 <pad> 在法语词汇表 FR_TEXT.vocab 中的索引值。

* (target_seq != target_pad)：产生一个布尔张量，其形状与 target_seq 相同，用于判断每个位置上的法语单词是否等于填充标记 <pad>。如果某个位置的单词不是 <pad>，则对应位置上的值为 True，否则为 False。

* .unsqueeze(1)：在第 1 维度上增加一个维度，将布尔张量的形状从 [sequence_length, batch_size] 变为 [sequence_length, 1, batch_size]。

* np.triu(np.ones(1, size, size), k=1).astype('uint8')：使用 NumPy 创建一个上三角矩阵，矩阵的形状为 (1, size, size)，其中 size 是目标序列的长度。该矩阵的上三角部分元素为 1，下三角部分和对角线元素为 0。

* nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0)：将上述创建的上三角矩阵转换为 PyTorch 张量，并将其中的非零元素设为 False，零元素设为 True。这样得到的 nopeak_mask 张量与目标序列长度相关。

* target_msk = target_msk & nopeak_mask：对目标掩码张量 target_msk 和 nopeak_mask 进行逐元素的逻辑与操作。这个操作的目的是将目标掩码中对应位置的元素，与 nopeak_mask 对应位置的元素进行逻辑与操作，以实现去除未来信息的效果。即，在目标掩码中，将未来位置的元素标记为 False，过滤掉模型不应该看到的未来信息。

具体来说，np.triu 函数的语法为：
    ```
python
np.triu(m, k=0)
    ```
参数说明：

m：要提取上三角部分的矩阵。
k（可选）：表示对角线的偏移量，即从主对角线向上偏移的行数。默认值为 0，表示不偏移。

## 多头注意力
一旦我们有了我们的嵌入值（带有位置编码）和我们的掩码，我们就可以开始构建我们模型的层。
![image](https://github.com/Mrgengli/pytroch_learning/blob/main/multihead.png?raw=true)
V、K 和 Q 分别代表“键”、“值”和“查询”。这些是注意力函数中使用的术语，但老实说，我不认为解释这些术语对于理解模型特别重要。

在编码器的情况下，V、K和G将只是嵌入向量的相同副本（加上位置编码）。它们的尺寸为 Batch_size * seq_len * d_model。

在多头注意力中，我们将嵌入向量分成N个头，因此它们将具有 batch_size * N * seq_len * (d_model / N) 的维度。

我们将这个最终维度 (d_model / N) 称为 d_k。

让我们看看解码器模块的代码：
```
class MultiHeadAttention(nn.Module): 
    def __init__(self, heads, d_model, dropout = 0.1): 
        super().__init__() 
        
        self.d_model = d_model 
        self.d_k = d_model // heads 
        self.h = heads 
        
        self.q_linear = nn.Linear(d_model, d_model) 
        self.v_linear = nn.Linear(d_model, d_model) 
        self.k_linear = nn.Linear(d_model, d_model) 
        self.dropout = nn.Dropout(dropout) 
        self.out = nn.Linear (d_model, d_model) 
    
    def forward(self, q, k, v, mask=None): 
        
        bs = q.size(0) 
        
        # 执行线性运算并分成h个头
        # 使用view()修改的是原始张量，使用reshape（）会生成一个新的张量
        k = self.k_linear(k).view(bs , -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k) 
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k) 
       
        # 转置获得尺寸 bs * h * sl * d_model 
        # 对1，2维度的张量进行转换
        k = k.transpose(1,2) 
        q = q.transpose(1,2) 
        v = v.transpose(1,2)
    
        # 使用函数计算注意力，我们将定义下一个
        scores = attention(q, k, v, self.d_k, mask, self.dropout) 
        
        # 连接头部并通过最终线性层
        # contiguous() 方法可以检查一个张量是否是的内存地址连续的，并在必要时重新组织内存中的数据，以使得张量变成连续的。
        # 在一开始创建一个张量的时候，张量内部的地址是连续的，但是经过后续的各种操作之后可能不是连续的。
        concat = scores.transpose(1,2).contiguous ().view(bs, -1, self.d_model) 
        
        output = self.out(concat)
    
        return output
```
## 注意力函数
![image](https://github.com/Mrgengli/pytroch_learning/blob/main/attention.png?raw=true)
    
另一个未显示的步骤是 dropout，我们将在 Softmax 之后应用它。

最后，最后一步是在到目前为止的结果和 V 之间进行点积。

这是注意功能的代码：
```
def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
if mask is not None:
        mask = mask.unsqueeze(1)
        #参数 mask == 0 是一个布尔类型的 PyTorch 张量，其中元素为 True 表示对应位置的元素需要进行填充操作。后面的为要填充的值
        scores = scores.masked_fill(mask == 0, -1e9)
    
scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)  #？？？？？z怎么调用的
        
    output = torch.matmul(scores, v)
    return output
```
## 规范化
归一化在深度神经网络中非常重要。它可以防止层中值的范围变化太大，这意味着模型训练更快并且具有更好的泛化能力。
```
class Norm(nn.Module): 
    def __init__(self, d_model, eps = 1e-6): 
        super().__init__() 
    
        self.size = d_model 
        # 创建两个可学习的参数来校准归一化
        self.alpha = nn.Parameter( torch.ones(self.size)) 
        self.bias = nn.Parameter(torch.zeros(self.size)) 
        self.eps = eps 
    def forward(self, x): 
        norm = self.alpha * (x - x. mean(dim=-1, keepdim=True)) \ 
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
```
## 模型搭建
![image](https://github.com/Mrgengli/pytroch_learning/blob/main/trans_model.png?raw=true)
    
上图中的编码器和解码器代表一层编码器和一层解码器。N 是层数的变量。例如。如果 N=6，数据经过六个编码器层（如上所示的架构），然后这些输出被传递到解码器，解码器也由六个重复的解码器层组成。

我们现在将使用上面模型中显示的架构构建 EncoderLayer 和 DecoderLayer 模块。然后当我们构建编码器和解码器时，我们可以定义有多少层。 

```
# 构建一个带有一个多头注意力层和一个前馈层的编码器层
class EncoderLayer(nn.Module): 
    def __init__(self, d_model, heads, dropout = 0.1): 
        super().__init__() 
        self.norm_1 = Norm(d_model) 
        self.norm_2 = Norm(d_model) 
        self.attn = MultiHeadAttention (heads, d_model) 
        self.ff = FeedForward(d_model) 
        self.dropout_1 = nn.Dropout(dropout) 
        self.dropout_2 = nn.Dropout(dropout) 
        
    def forward(self, x, mask): 
        x2 = self.norm_1(x ) 
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask)) 
        x2 = self.norm_2(x) 
        x = x + self.dropout_2(self.ff(x2)) 
        return x 
    
# 建立一个具有两个多头注意层的解码器层和
# 一个前馈层
class DecoderLayer(nn.Module): 
    def __init__(self, d_model, heads, dropout=0.1): 
        super().__init__() 
        self.norm_1 = Norm(d_model) 
        self.norm_2 = Norm(d_model) 
        self.norm_3 = Norm (d_model) 
        
        self.dropout_1 = nn.Dropout(dropout) 
        self.dropout_2 = nn.Dropout(dropout) 
        self.dropout_3 = nn.Dropout(dropout) 
        
        self.attn_1 = MultiHeadAttention(头, d_model) 
        self.attn_2 = MultiHeadAttention(头, d_model) 
        self.ff = FeedForward(d_model).cuda()
def forward(self, x, e_outputs, src_mask, trg_mask): 
        x2 = self.norm_1(x) 
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask)) 
        x2 = self.norm_2(x ) 
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, 
        src_mask)) 
        x2 = self.norm_3(x) 
        x = x + self.dropout_3(self.ff(x2))
        return x
# 然后我们可以构建一个可以生成多层的方便的克隆函数：
def get_clones(module, N): 
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)]) 
```
## 我们现在准备构建编码器和解码器：
```
class Encoder（nn.Module）：
    def __init__（self，vocab_size，d_model，N，heads）：
        super（）.__init__（）
        self.N = N 
        self.embed = Embedder（vocab_size，d_model）
        self.pe = PositionalEncoder（ d_model) 
        self.layers = get_clones(EncoderLayer(d_model, heads), N) 
        self.norm = Norm(d_model) 
    def forward(self, src, mask): 
        x = self.embed(src) 
        x = self.pe(x ) 
        for i in range(N): 
            x = self.layers[i](x, mask) 
        return self.norm(x) 
    
class Decoder(nn.Module): 
    def __init__(self, vocab_size, d_model, N, heads) : 
        super().__init__() 
        self.N = N
        self.embed = Embedder(vocab_size, d_model) 
        self.pe = PositionalEncoder(d_model) 
        self.layers = get_clones(DecoderLayer(d_model, heads), N) 
        self.norm = Norm(d_model) 
    def forward(self, trg, e_outputs, src_mask, trg_mask): 
        x = self.embed(trg) 
        x = self.pe(x) 
        for i in range(self.N): 
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)
```
# The transformer!
```
class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads)
        self.decoder = Decoder(trg_vocab, d_model, N, heads)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output 
    # 我们不对输出执行 softmax，因为这将由我们的损失函数自动处理 
```    
    
## 训练模型
构建好转换器后，剩下的就是在 EuroParl 数据集上训练那个笨蛋。编码部分非常轻松，但请准备好等待大约 2 天让该模型开始收敛！
    
* 首先来定义一些参数
```
d_model = 512
heads = 8
N = 6
src_vocab = len(EN_TEXT.vocab)
trg_vocab = len(FR_TEXT.vocab)
model = Transformer(src_vocab, trg_vocab, d_model, N, heads)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
# this code is very important! It initialises the parameters with a
# range of values that stops the signal fading or getting too big.
# See [this blog](https://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization) for a mathematical explanation.
optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
```
## train
```
def train_model(epochs, print_every=100): 
    
    model.train() 
    
    start = time.time() 
    temp = start 
    
    total_loss = 0 
    
    for epoch in range(epochs):
       
        for i, batch in enumerate(train_iter):
            src = batch.English.transpose(0,1) 
            trg = batch.French.transpose(0,1)
            # 我们输入的法语句子包含除
            最后一个以外的所有词，因为它使用每个词来预测下一个词
            
            trg_input = trg[:, :-1] 
            
            # 我们试图预测的词
            
            targets = trg[:, 1: ].contiguous().view(-1) 
            
            # 创建函数以使用上面的掩码代码制作掩码
            
            src_mask, trg_mask = create_masks(src, trg_input) 
            
            preds = model(src, trg_input, src_mask, trg_mask) 
            
            optim.zero_grad() 
            
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)),
            results, ignore_index=target_pad)
            loss.backward() 
            optim.step() 
            
            total_loss += loss.data[0] 
            if (i + 1) % print_every == 0: 
                loss_avg = total_loss / print_every
                print("time = %dm, epoch %d, iter = %d, loss = %.3f, 
                %ds per %d iters" % ((time.time() - start) // 60, 
                epoch + 1, i + 1, loss_avg, time.time() - temp, 
                print_every )) 
                total_loss = 0 
                temp = time.time()
```
## 训练结果
![image](https://github.com/Mrgengli/pytroch_learning/blob/main/train_result.png?raw=true)
    
## 测试模型
我们可以使用下面的函数来翻译句子。我们可以直接从我们的批次中输入句子，或输入自定义字符串。

翻译器通过运行一个循环来工作。我们从对英文句子进行编码开始。然后我们将 <sos> 标记索引和编码器输出提供给解码器。解码器对第一个单词进行预测，我们将其添加到带有 sos 令牌的解码器输入中。我们重新运行循环，获取下一个预测并将其添加到解码器输入中，直到我们到达 <eos> 标记，让我们知道它已完成翻译。
```    
def translate(model, src, max_len = 80, custom_string=False):
    
    model.eval()
if custom_sentence == True:
        src = tokenize_en(src)
        sentence=\
        Variable(torch.LongTensor([[EN_TEXT.vocab.stoi[tok] for tok
        in sentence]])).cuda()
src_mask = (src != input_pad).unsqueeze(-2)
    e_outputs = model.encoder(src, src_mask)
    
    outputs = torch.zeros(max_len).type_as(src.data)
    outputs[0] = torch.LongTensor([FR_TEXT.vocab.stoi['<sos>']])
for i in range(1, max_len):    
            
        trg_mask = np.triu(np.ones((1, i, i),
        k=1).astype('uint8')
        trg_mask= Variable(torch.from_numpy(trg_mask) == 0).cuda()
        
        out = model.out(model.decoder(outputs[:i].unsqueeze(0),
        e_outputs, src_mask, trg_mask))
        out = F.softmax(out, dim=-1)
        val, ix = out[:, -1].data.topk(1)
        
        outputs[i] = ix[0][0]
        if ix[0][0] == FR_TEXT.vocab.stoi['<eos>']:
            break
return ' '.join(
    [FR_TEXT.vocab.itos[ix] for ix in outputs[:i]]
    ) 
```   
    
