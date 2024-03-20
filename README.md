# Griffin
Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models

[arXiv](https://arxiv.org/abs/2402.19427)

# Model Architecture
All our models contain the following components: (i) a residual block, (ii) an MLP block, and (iii) a temporal-mixing block. While (i) and (ii) are the same across all models, we consider three temporal mixing blocks: global Multi-Query Attention (MQA), local (sliding-window) MQA and our proposed recurrent block. As part of the recurrent block we use the Real-Gated Linear Recurrent Unit (RG-LRU) – a novel recurrent layer inspired by the Linear Recurrent Unit [Orvieto et al., 2023b](https://arxiv.org/abs/2303.06349).

The residual block, as shown in Figure 2(a), defines the global structure of our models and is inspired by pre-norm Transformers (Xiong et al., 2020). After embedding the input sequence we pass it through $N$ such blocks ($N$ denoting the model depth), and then we apply RMSNorm [Zhang and Sennrich, 2019](https://arxiv.org/abs/1910.07467) to produce the final activations. To compute the token probabilities we apply a final linear layer followed by a softmax. The weights of this layer are shared with the input embedding layer.
## Residual block
![Griffin](https://arxiv.org/html/2402.19427v1/x3.png)
Figure 2: a) The main backbone of our mode architecture is the residual block, which is stacked $N$ times. b) The gated MLP block that we use. c) The recurrent block that we propose as an alternative to Multi Query Attention (MQA). It uses our proposed RG-LRU layer, defined in Section 2.4.

The residual block contains two components, applied in order. The first component takes the hidden state $\chi$ and applies an RMSNorm [Zhang and Sennrich, 2019](https://arxiv.org/abs/1910.07467), followed by the temporal-mixing block. We then merge the output with a skip connection from $\chi$ through addition. Similarly, the second component applies RMSNorm, followed by the MLP block and then merges its output with a skip connection from the input of the RMSNorm. This block is illustrated in Figure 2 (a).

## MLP block
We use a gated MLP block  [Dauphin et al., 2017](https://arxiv.org/abs/1612.08083) (illustrated in Figure 2(b)), which creates two branches from its input of dimension
$D$. We apply a linear layer with output dimension $MD$
 on each branch, where $M$ denotes the expansion factor. For simplicity, we use $M=3$ throughout this work. We apply a GeLU non-linearity [Hendrycks and Gimpel, 2016](https://arxiv.org/abs/1606.08415) on one of the branches before merging them by element-wise multiplication, similar to GeGeLU [Shazeer, 2020](https://arxiv.org/abs/2002.05202). However, in our MLP block, we apply a final linear layer with output dimension $D$ on the outputs of the GeGeLU layer.

## Temporal-mixing blocks
The temporal-mixing block is the component of our model that aggregates hidden layer activations at different temporal locations in the sequence. We consider three temporal-mixing blocks: global MQA [Shazeer, 2019](https://arxiv.org/abs/1911.02150), local MQA [Beltagy et al., 2020](https://arxiv.org/abs/2004.05150) and our proposed Recurrent block.

### Global multi-query attention
Unless otherwise stated, we use MQA rather than MHA to improve the inference speeds of our Transformer baselines [Shazeer, 2019](https://arxiv.org/abs/1911.02150). We use a fixed head dimension $D_{head}=128$, and we fix the number of attention heads $H$ such that $HD_{head}=D$. This requires the model dimension $D$ to be a multiple of 128. We do not use any absolute positional embeddings, but we use Rotary Position Embedding (RoPE) [Su et al., 2021](https://arxiv.org/abs/2104.09864) as a relative positional embedding.

### Local sliding window attention
