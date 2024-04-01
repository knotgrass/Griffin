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
One of the key disadvantages of using global attention is that its computational complexity grows quadratically in the sequence length. To address this, several works have started to adopt local attention [Beltagy et al., 2020](https://arxiv.org/abs/2004.05150), also known as sliding window attention. It allows each position to attend only to a fixed number of tokens in the past. This not only reduces the computational FLOPs but also bounds the size of the KV cache to the size of window, making it no longer quadratic in the sequence length. All other details are the same as the global MQA.

### Recurrent block
Our recurrent block (Figure 2(c)) is similar to the GSS block [Mehta et al., 2022](https://arxiv.org/abs/2206.13947) and the block used by Mamba [Gu and Dao, 2023](https://arxiv.org/abs/2312.00752). We take the input of dimension $D$  and apply two linear layers with output dimension $D_{RNN}$ in parallel, creating two branches. On the first branch, we apply a small separable Conv1D layer, inspired by the Shift-SSM in H3 [Dao et al., 2022b](https://arxiv.org/abs/2212.14052), with a temporal filter dimension of 4. Note that this Conv1D layer is very small, with just $4D$
 parameters. We follow the Conv1D layer with our proposed RG-LRU layer (defined below.) On the second branch we apply a GeLU nonlinearity and then merge the branches by element-wise multiplication. We then apply a final linear layer with output dimension $D$.

## Real-Gated Linear Recurrent Unit (RG-LRU)
Our proposed RG-LRU layer has a simple recurrence inspired by the Linear Recurrent Unit (LRU) [Orvieto et al., 2023b](https://arxiv.org/abs/2303.06349), but incorporates a gating mechanism motivated by the literature on non-linear RNNs, in particular LSTMs [Hochreiter and Schmidhuber, 1997](https://direct.mit.edu/neco/article-abstract/9/8/1735/6109/Long-Short-Term-Memory?redirectedFrom=fulltext) and GRUs [Chung et al., 2014](https://arxiv.org/abs/1412.3555). The equations describing the layer are as follows:

$$\begin{align}
r_t &= \sigma(W_{a} x_t + b_a), & \text{recurrence gate} \\
i_t &= \sigma(W_{x} x_t + b_x), & \text{input gate} \\
a_t &= a^{cr_t}, & \text{} \\
h_t &= a_t \odot h_{t-1} + \sqrt{1 - a_t^2} \odot (i_t \odot x_t). & \text{}
\end{align}$$

The output of the layer is $y_t=h_t$, and the non-linearity $\sigma$ in the equations is the sigmoid function. The recurrent weight $a$ in Equation (4) is diagonal. Hence all operations are element-wise. We parameterize $a$ in Equation (3) as $a=\sigma(\Lambda)$, where $\Lambda$ is a learnable parameter. This guarantees that $0 <= a <= 1$, ensuring that the recurrence is stable. The variable $c$ is a scalar-valued constant set to 8. For numerical stability, in practice we compute $a^{cr_t}$ in log-space (see Appendix A). The layer has gates on both the input $x$ and the recurrent weight $a$. However, neither gate depends on the recurrent state $h_{t-1}$, which ensures that the computation can be executed efficiently on device. We initialize both $W_{a}$ and $W_{b}$ using LeCun init [LeCun et al., 2002](https://cseweb.ucsd.edu/classes/wi08/cse253/Handouts/lecun-98b.pdf). We initialize $\Lambda$ such that $a^c$ is uniformly distributed between $0.9$ and $0.999$ at the start of training, similar to ([Orvieto et al., 2023b](https://arxiv.org/abs/2303.06349).). Unlike many recent works in the SSM literature, the RG-LRU does not use initialization inspired by the theory of orthogonal polynomials [Gu et al., 2020](https://proceedings.neurips.cc/paper/2020/hash/102f0bb6efb3a6128a3c750dd16729be-Abstract.html), and it also is not defined as the discretization of an underlying continuous system [Gu et al., 2021a](https://arxiv.org/abs/2111.00396). Unlike the original LRU layer, we do not use complex algebra in the recurrence. While using complex recurrences would lead to a more expressive layer [Orvieto et al., 2023a](https://arxiv.org/abs/2307.11888) we found that complex recurrences were not beneficial for language modelling in practice, as also observed by [Gu and Dao, 2023](https://arxiv.org/abs/2312.00752). (see Appendix B)

### Gate behaviour
The input gate $i_t$ is similar to the one in LSTM, which can filter (or scale down) the input $x_t$
. However, to our knowledge, our recurrence gate $r_t$ is different from other gating mechanisms in the literature. For example, the selection mechanism proposed in Mamba [Gu and Dao, 2023](https://arxiv.org/abs/2312.00752) is comparable to the update gate of GRUs which interpolates  $x_t$. Its effect on the hidden state allows it to reset its state and forget any information it holds from the past, similar to the forget gate in the LSTM. In contrast, our recurrence gate can approximately interpolate between the standard LRU update from [Orvieto et al., 2023a](https://arxiv.org/abs/2307.11888) and the previous hidden state, which allows it to effectively discard the input and preserve all information from the previous history (see Appendix A for further details). We believe the key role of this gate is to enable the model to achieve super-exponential memory by reducing the influence of uninformative inputs.
