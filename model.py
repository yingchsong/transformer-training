import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import copy
import math

# 深拷贝
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# 未来位置掩码
def subsequent_mask(size):
    "Mask out subsequent positions."

    attn_shape = (1, size, size)  # 步骤1：定义掩码矩阵的形状
    # 步骤2：生成上三角矩阵（对角线以上为1，对角线及以下为0）
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    # 步骤3：反转掩码（上三角1→0，对角线及下三角0→1），最终得到“允许关注”的位置
    return subsequent_mask == 0



# 归一化模块
class LayerNorm(nn.Module):
    def __init__(self,features, eps=1e-6):
        super(LayerNorm,self).__init__()
        # 可学习的缩放参数（a_2）：初始化为全1，形状 [features]
        self.a_2 = nn.Parameter(torch.ones(features))
        # 可学习的偏移参数（b_2）：初始化为全0，形状 [features]
        self.b_2 = nn.Parameter(torch.zeros(features))
        # 极小值eps：防止分母为0（除以std时），默认1e-6
        self.eps = eps

    def forward(self,x):
        # 对张量的最后一个维度
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# 连接模块
class SublayerConnection(nn.Module):
    def __init__(self,size,dropout):
        super(SublayerConnection,self).__init__()
        # 初始化层归一化模块：size对应d_model（特征维度，论文中512）
        self.norm = LayerNorm(size)
        # 初始化Dropout层：防止过拟合，默认dropout概率一般为0.1
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,sublayer):
        # 核心逻辑：残差连接 = 原始输入x + 子层输出（经dropout）
        # 步骤拆解：
        # 1. self.norm(x)：先对输入x做层归一化，稳定数值分布
        # 2. sublayer(...)：执行子层操作（比如自注意力层/前馈网络层）
        # 3. self.dropout(...)：对子层输出做dropout，防止过拟合
        # 4. x + ...：残差连接（原始输入 + 子层处理后的输出）
        return x + self.dropout(sublayer(self.norm(x)))

# Feedward network
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

# Encoder parts
class Encoder(nn.Module):
    """
        Transformer完整编码器：由N个相同的EncoderLayer堆叠而成
        核心逻辑：源序列特征 → 逐层经过N个EncoderLayer → 层归一化 → 最终语义表征
    """
    def __init__(self,layer,N):
        super(Encoder,self).__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)
    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)

class EncoderLayer(nn.Module):
    def __init__(self,size,self_attn,feed_forward,dropout):
        super(EncoderLayer,self).__init__()

        self.self_attn = self_attn                  # 多头自注意力子层
        self.feed_forward = feed_forward            # 前馈网络子层
        # 创建两个SublayerConnection，分别适配自注意力和前馈网络
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    def forward(self,x,mask):
        # 第一个子层：自注意力层 + 残差连接
        x = self.sublayer[0](x, lambda x : self.self_attn(x,x,x,mask))
        # 第二个子层：前馈网络层 + 残差连接
        return self.sublayer[1](x,self.feed_forward)

# Decoder parts
class Decoder(nn.Module):
    def __init__(self,layer,N):
        super(Decoder,self).__init__()

        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        解码器前向传播：逐层处理目标序列特征，结合源语义记忆和掩码
        Args:
            x (torch.Tensor): 目标序列特征（嵌入+位置编码），形状 [batch_size, tgt_seq_len, d_model]
            memory (torch.Tensor): 编码器输出的源序列语义记忆，形状 [batch_size, src_seq_len, d_model]
            src_mask (torch.Tensor): 源序列掩码，遮挡padding，形状 [batch_size, 1, src_seq_len]
            tgt_mask (torch.Tensor): 目标序列掩码，遮挡padding+未来位置，形状 [batch_size, tgt_seq_len, tgt_seq_len]
        Returns:
            torch.Tensor: 解码器最终输出的目标特征，形状 [batch_size, tgt_seq_len, d_model]
                          需传入Generator层生成词表概率
         """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size                        # 特征维度（d_model，论文中512），用于层归一化和残差连接的维度校验
        self.self_attn = self_attn              # 掩码自注意力层（关注目标序列自身）
        self.src_attn = src_attn                # 编码器-解码器注意力层（跨注意力，关注源序列memory）
        self.feed_forward = feed_forward        # 前馈网络层（特征变换）
        # 创建3个SublayerConnection：分别适配3个子层，每个都带残差+归一化+dropout
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        # 第一步：掩码自注意力子层（第0个SublayerConnection）
        # 输入：目标序列x + 目标掩码tgt_mask（遮挡未来位置+padding）
        # 作用：目标序列只能关注自身“当前及之前位置”的特征，避免信息泄露
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 第二步：编码器-解码器注意力子层（第1个SublayerConnection）
        # 输入：更新后的x + 源序列memory + 源掩码src_mask（遮挡源序列padding）
        # 作用：让目标序列特征关注源序列的对应位置（比如翻译时“我”对应“I”）
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # 第三步：前馈网络子层（第2个SublayerConnection）
        # 作用：对每个位置的特征做独立的线性变换，提升模型表达能力
        return self.sublayer[2](x, self.feed_forward)

# Attention Parts

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # 步骤1：获取Query的最后一维维度d_k（对应论文中的d_k）
    # query.size(-1)：取最后一维的大小，兼容任意维度输入（比如[batch, n_head, seq_len, d_k]）
    d_k = query.size(-1)
    # 步骤2：计算Query和Key的点积（矩阵乘法实现批量点积） + 缩放
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 步骤3：应用掩码（mask）——遮挡无效位置（padding/未来位置）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # 步骤4：对得分矩阵最后一维做softmax，得到注意力权重
    p_attn = scores.softmax(dim=-1)
    # 步骤5：应用dropout（可选）——防止过拟合
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 返回两个值：输出张量 + 注意力权重矩阵（p_attn可用于查看模型关注了哪些位置）
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        #模型维度d_model和头数h
        super(MultiHeadedAttention,self).__init__()

        assert d_model % h == 0 # 断言：d_model必须能被h整除（保证每个头的维度d_k是整数）

        self.d_k = d_model // h  # 每个头的维度
        self.h = h # 头数
        self.linears = clones(nn.Linear(d_model, d_model), 4) # 4个独立的Linear层（参数不共享）：
        # - 前3个：分别用于Q/K/V的投影（W_i^Q/W_i^K/W_i^V）
        # - 最后1个：用于多头拼接后的输出投影（W^O）
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # 掩码扩展维度：mask.shape [batch, seq_len] → [batch, 1, seq_len]
            # 目的：让同一个掩码应用到所有h个头（通过广播匹配多头维度）
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)  # 获取批次大小（batch_size）
        # 1) Do all the linear projections in batch from d_model => h x d_k
        #线性投影+维度转化
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
            #zip迭代器匹配，将输入的两个元素进行匹配，第二个元素有三个(query, key, value)
            #(self.linears[0], query)
            #(self.linears[1], key)
            #(self.linears[2], value)
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)           # 交换维度1和2 → [batch, seq_len, h, d_k]
            .contiguous()                           # 整理内存（避免view失败）
            .view(nbatches, -1, self.h * self.d_k)  # 拼接h个头 → [batch, seq_len, h×d_k=d_model]
        )

        del query
        del key
        del value
        # 最终线性投影（W^O）：将拼接后的结果投影回d_model维度，返回最终输出
        return self.linears[-1](x)

# Embedding parts
class Embeddings(nn.Module):
    def __init__(self,d_model, vocab):
        super(Embeddings,self).__init__()
        # - vocab：词汇表大小（比如中文词汇表有10000个词）
        # - d_model：嵌入向量维度（论文中512）
        # - 输出：将词索引映射为d_model维的连续向量（可学习参数）
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model # 保存嵌入维度，用于后续缩放

    def forward(self,x):
        # 步骤1：self.lut(x) → 将词索引x映射为嵌入向量，形状 [batch, seq_len, d_model]
        # 步骤2：* math.sqrt(self.d_model) → 对嵌入向量做缩放（论文关键设计）
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding,self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        # 步骤1：初始化位置编码矩阵pe，形状[max_len, d_model]（max_len=5000表示支持最长5000个token的序列）
        pe = torch.zeros(max_len, d_model)
        # 步骤2：生成位置索引pos，形状[max_len, 1]（从0到4999的位置）
        # unsqueeze(1)：将[max_len]转为[max_len,1]，方便后续和div_term做广播乘法
        position = torch.arange(0, max_len).unsqueeze(1)
        # 步骤3：计算分母项div_term（对应公式中的10000^(2i/d_model)，用对数+指数转换避免数值溢出）
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 步骤4：填充正弦/余弦值到pe矩阵（对应论文公式）
        # pe[:, 0::2]：所有行，从0开始、步长2的列（偶数维度）→ 填充sin值
        pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2]：所有行，从1开始、步长2的列（奇数维度）→ 填充cos值
        pe[:, 1::2] = torch.cos(position * div_term)
        # 步骤5：扩展batch维度，形状[max_len, d_model] → [1, max_len, d_model]
        # 目的：支持广播（后续和[batch, seq_len, d_model]的词嵌入相加）
        pe = pe.unsqueeze(0)
        # 步骤6：将pe注册为缓冲区（buffer）
        # 含义：pe是不参与梯度更新的参数（位置编码固定，无需学习），但会随模型保存/加载
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x：输入的词嵌入向量，形状[batch, seq_len, d_model]
        # self.pe[:, :x.size(1)]：截取前x.size(1)个位置的编码（适配当前序列长度），形状[1, seq_len, d_model]
        # requires_grad_(False)：明确指定不计算梯度（双重保障）
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class Generator(nn.Module):
    def __init__(self,d_model,vocab):
        # d_model：解码器输出特征的维度（论文中为512）
        # vocab：目标词表的大小（比如翻译任务中目标语言的总词数）
        super(Generator,self).__init__()
        # 定义线性投影层：将d_model维特征映射到词表维度
        self.proj = nn.Linear(d_model, vocab)

    def forward(self,x):
        # 1. self.proj(x)：线性层将特征从d_model维 → vocab维
        # 2. log_softmax(..., dim=-1)：对最后一维做log_softmax，将数值转为对数概率
        return log_softmax(self.proj(x), dim=-1)


class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder,src_embed, tgt_embed, generator):
        super(EncoderDecoder,self).__init__()
        self.encoder = encoder  # 编码器模块
        self.decoder = decoder  # 解码器模块
        self.src_embed = src_embed  # 源语言嵌入层（将源词转为向量）
        self.tgt_embed = tgt_embed  # 目标语言嵌入层（将目标词转为向量）
        self.generator = generator  # 生成器模块（输出词表概率）

    def forward(self,src, tgt, src_mask, tgt_mask):

        self.memory = self.encoder(src,src_mask)
        return self.decoder(self.memory, src_mask, tgt, tgt_mask)

    def encode(self,src,src_mask):
        """
            对源序列进行编码，生成源序列的语义记忆（memory）
            核心逻辑：源序列token → 词嵌入向量 → 编码器编码（结合掩码）

            Args:
                src (torch.Tensor): 源序列token张量，形状一般为 [batch_size, src_seq_len]
                src_mask (torch.Tensor): 源序列掩码张量，用于遮挡padding部分，形状 [batch_size, 1, src_seq_len]

            Returns:
                torch.Tensor: 编码器输出的语义记忆（memory），形状 [batch_size, src_seq_len, d_model]
                              包含源序列的全部语义信息，供解码器使用
        """
        return self.encoder(self.src_embed(src),src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
            基于编码器的语义记忆对目标序列进行解码，生成目标序列的特征表示
            核心逻辑：目标序列token → 词嵌入向量 → 解码器解码（结合memory和各类掩码）

            Args:
                memory (torch.Tensor): 编码器输出的源序列语义记忆，形状 [batch_size, src_seq_len, d_model]
                src_mask (torch.Tensor): 源序列掩码张量，用于解码器的跨注意力层，避免关注源序列padding
                tgt (torch.Tensor): 目标序列token张量，形状一般为 [batch_size, tgt_seq_len]
                tgt_mask (torch.Tensor): 目标序列掩码张量，遮挡padding和未来位置，形状 [batch_size, tgt_seq_len, tgt_seq_len]

            Returns:
                torch.Tensor: 解码器输出的目标序列特征，形状 [batch_size, tgt_seq_len, d_model]
                              需传入Generator层才能生成词表概率
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

