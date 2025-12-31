# transformer-training




```sql
Transformer整体架构
├─ 工具函数（无类依赖，基础能力）
│  ├─ clones(深拷贝模块) ── 被MultiHeadedAttention、EncoderLayer、DecoderLayer、Encoder、Decoder依赖
│  ├─ subsequent_mask(未来位置掩码) ── 被EncoderDecoder（推理函数）依赖
│  └─ attention(缩放点积注意力) ── 被MultiHeadedAttention依赖
│
├─ 基础组件层（所有核心模块的基础）
│  ├─ LayerNorm(层归一化)
│  │  ├─ 被SublayerConnection依赖
│  │  ├─ 被Encoder依赖
│  │  ├─ 被Decoder依赖
│  │  └─ 被EncoderLayer/DecoderLayer间接依赖
│  │
│  ├─ SublayerConnection(残差连接+归一化+Dropout)
│  │  ├─ 被EncoderLayer依赖
│  │  └─ 被DecoderLayer依赖
│  │
│  ├─ PositionwiseFeedForward(位置前馈网络)
│  │  ├─ 被EncoderLayer依赖
│  │  └─ 被DecoderLayer依赖
│  │
│  ├─ Embeddings(词嵌入+缩放) ── 被EncoderDecoder依赖
│  │
│  ├─ PositionalEncoding(位置编码) ── 被EncoderDecoder依赖
│  │
│  └─ Generator(输出层：线性+log_softmax) ── 被EncoderDecoder依赖
│
├─ 核心子层模块（基础组件组装）
│  ├─ MultiHeadedAttention(多头注意力)
│  │  ├─ 依赖：clones、attention、nn.Linear
│  │  ├─ 被EncoderLayer依赖（自注意力）
│  │  └─ 被DecoderLayer依赖（掩码自注意力+跨注意力）
│  │
│  ├─ EncoderLayer(编码器单层)
│  │  ├─ 依赖：MultiHeadedAttention、PositionwiseFeedForward、SublayerConnection
│  │  └─ 被Encoder依赖
│  │
│  ├─ Encoder(编码器，N层堆叠)
│  │  ├─ 依赖：EncoderLayer、LayerNorm、clones
│  │  └─ 被EncoderDecoder依赖
│  │
│  ├─ DecoderLayer(解码器单层)
│  │  ├─ 依赖：MultiHeadedAttention、PositionwiseFeedForward、SublayerConnection
│  │  └─ 被Decoder依赖
│  │
│  └─ Decoder(解码器，N层堆叠)
│     ├─ 依赖：DecoderLayer、LayerNorm、clones
│     └─ 被EncoderDecoder依赖
│
└─ 整体模型层
   └─ EncoderDecoder(Transformer主类)
      ├─ 依赖：Encoder、Decoder、Embeddings、PositionalEncoding、Generator
      └─ 对外提供：encode(编码源序列)、decode(解码目标序列)、forward(前向传播)
```
