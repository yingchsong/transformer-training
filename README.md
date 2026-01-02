# Transformer 模型实现说明文档

本项目基于 PyTorch 实现了经典的 **Transformer** 架构（[Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)），包含完整的编码器（Encoder）、解码器（Decoder）、多头注意力机制、位置编码、嵌入层和生成器模块。以下将详细说明从输入到输出的整个流程，以及各组件的输入/输出结构。

---

## 📌 整体架构概览

```text
[Source Tokens] ──(src_embed + pos_enc)──► Encoder ──► Memory (Context)
                                                           │
[Tgt Tokens]   ──(tgt_embed + pos_enc)──► Decoder ◄───────┘
                                                           │
                                                    Generator (Linear + log_softmax)
                                                           ↓
                                                [Log-probabilities over vocab]
```

模型由 `EncoderDecoder` 类统一管理，其前向传播接口为：

```python
output = model(src, tgt, src_mask, tgt_mask)
```

---

## 🔤 输入说明

| 输入项       | 张量形状                     | 含义 |
|--------------|------------------------------|------|
| `src`        | `[batch_size, src_seq_len]`  | 源序列（如英文句子）的 token ID |
| `tgt`        | `[batch_size, tgt_seq_len]`  | 目标序列（如中文翻译）的 token ID（训练时为右移一位的 ground truth） |
| `src_mask`   | `[batch_size, 1, src_seq_len]` | 源序列 padding 掩码（1 表示有效 token，0 表示 padding） |
| `tgt_mask`   | `[batch_size, tgt_seq_len, tgt_seq_len]` | 目标序列掩码：结合 padding 掩码 + **未来位置掩码**（防止信息泄露） |

> 💡 `tgt_mask` 通常由 `subsequent_mask(tgt_seq_len)` 与 padding 掩码按位“与”得到。

---

## 🧠 编码器（Encoder）

### 结构组成
- 由 `N` 个相同的 `EncoderLayer` 堆叠而成。
- 每个 `EncoderLayer` 包含：
  1. **多头自注意力子层**（Multi-Head Self-Attention）
  2. **前馈网络子层**（Position-wise Feed-Forward Network）
- 每个子层后接 **残差连接 + LayerNorm**（通过 `SublayerConnection` 实现）。
- 最终输出前再进行一次 `LayerNorm`。

### 输入
- **词嵌入 + 位置编码**：  
  `x = src_embed(src) + positional_encoding`  
  形状：`[batch_size, src_seq_len, d_model]`
- **源掩码 `src_mask`**：用于屏蔽 padding 位置。

### 输出
- **Memory（语义记忆）**：  
  形状：`[batch_size, src_seq_len, d_model]`  
  包含整个源序列的上下文感知表征，供 Decoder 的 cross-attention 使用。

---

## 🔍 解码器（Decoder）

### 结构组成
- 由 `N` 个相同的 `DecoderLayer` 堆叠而成。
- 每个 `DecoderLayer` 包含三个子层：
  1. **掩码多头自注意力**（Masked Multi-Head Self-Attention）  
     → 仅允许关注当前及之前的位置（防止未来信息泄露）
  2. **编码器-解码器注意力**（Multi-Head Cross-Attention）  
     → Query 来自 Decoder，Key/Value 来自 Encoder 的 memory
  3. **前馈网络**（Position-wise Feed-Forward Network）
- 每个子层后同样接 **残差连接 + LayerNorm**。
- 最终输出前进行一次 `LayerNorm`。

### 输入
- **目标嵌入 + 位置编码**：  
  `x = tgt_embed(tgt) + positional_encoding`  
  形状：`[batch_size, tgt_seq_len, d_model]`
- **Memory**：来自 Encoder 的输出
- **`src_mask`**：用于 cross-attention 中屏蔽源 padding
- **`tgt_mask`**：用于 self-attention 中屏蔽未来位置和目标 padding

### 输出
- **解码器特征表示**：  
  形状：`[batch_size, tgt_seq_len, d_model]`  
  每个位置的向量融合了：
  - 目标序列的历史信息（通过 masked self-attn）
  - 源序列的相关上下文（通过 cross-attn）

---

## 🎯 生成器（Generator）

- 一个简单的线性层 + `log_softmax`：
  ```python
  log_probs = log_softmax(linear(x), dim=-1)
  ```
- **输入**：Decoder 输出，`[batch_size, tgt_seq_len, d_model]`
- **输出**：对数概率分布，`[batch_size, tgt_seq_len, vocab_size]`

> ⚠️ 注意：训练时通常使用 `CrossEntropyLoss` 或 `NLLLoss`，因此输出为 **log-probabilities** 而非 raw logits。

---

## 📦 嵌入与位置编码

### `Embeddings`
- 将 token ID 映射为 `d_model` 维向量。
- **关键细节**：嵌入向量乘以 `√d_model`（论文建议，保持与位置编码相同量级）。

### `PositionalEncoding`
- 使用 **固定正弦/余弦函数** 编码位置信息（无需学习）。
- 支持最长 `max_len=5000` 的序列。
- 与词嵌入相加后经 Dropout 输出。

---

## 🧩 注意力机制

### `MultiHeadedAttention`
- 支持自注意力（self-attn）和交叉注意力（cross-attn）。
- 输入 Q/K/V 可来自不同来源（如 Decoder 中 Q 来自自身，K/V 来自 memory）。
- 内部调用 `attention()` 函数实现 **缩放点积注意力**（Scaled Dot-Product Attention）。
- 支持任意掩码（padding / future masking）。

---

## ✅ 使用示例（伪代码）

```python
# 初始化模型组件
encoder = Encoder(EncoderLayer(...), N=6)
decoder = Decoder(DecoderLayer(...), N=6)
src_embed = nn.Sequential(Embeddings(d_model, src_vocab), PositionalEncoding(...))
tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), PositionalEncoding(...))
generator = Generator(d_model, tgt_vocab)

model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)

# 前向传播
src = torch.randint(0, src_vocab, (32, 10))      # batch=32, src_len=10
tgt = torch.randint(0, tgt_vocab, (32, 8))       # tgt_len=8
src_mask = (src != PAD).unsqueeze(1)             # [32, 1, 10]
tgt_mask = make_std_mask(tgt)                    # [32, 8, 8] (含 future mask)

output = model(src, tgt, src_mask, tgt_mask)     # [32, 8, d_model]
log_probs = model.generator(output)              # [32, 8, tgt_vocab]
```

---

## 📚 总结

| 模块 | 输入 | 输出 | 关键作用 |
|------|------|------|--------|
| **Encoder** | `[B, S, d]`, `src_mask` | `[B, S, d]` | 提取源序列全局语义 |
| **Decoder** | `[B, T, d]`, `memory`, `src_mask`, `tgt_mask` | `[B, T, d]` | 自回归生成目标序列 |
| **Generator** | `[B, T, d]` | `[B, T, V]` | 映射到词表概率 |

> 其中：`B = batch_size`, `S = src_seq_len`, `T = tgt_seq_len`, `d = d_model`, `V = vocab_size`

本实现严格遵循原始论文结构，可作为教学、研究或自定义扩展的基础。

## 📖 参考资料

Rush, A. M. (2018). The annotated transformer. In Proceedings of Workshop for NLP Open Source Software (NLP-OSS) (pp. 52–60). Association for Computational Linguistics. https://aclanthology.org/W18-2509/