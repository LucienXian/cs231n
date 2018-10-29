# Recurrent Nerual Networks

## 目的

相比较于CNN，RNN能够更好地序列的信息。比如理解一句话的意思，根据视频输出行为，判断文本的情感等等。前后的输入是有关联的。

## 结构

RNN的结构中有一个内部状态值$h_t$，这个值取决于输入的x和上一次的内部状态值$h_{t-1}$。另外权重矩阵W随着时间的向前，是不会发生变化。同时每次的输入都会产生一个输出值$Y_t$

结构图如下：

![img](https://en.wikipedia.org/wiki/Recurrent_neural_network#/media/File:Recurrent_neural_network_unfold.svg)

RNN的公式如下：
$$
Y_t = g(V h_t) \\
h_t = f(UX_t+Wh_{t-1})
$$



## truncated backprop

假如数据序列很长，全部训练的话，我们无法全部放到RNN中，因为可能会造成梯度消失或者爆炸的问题，另外内存也容易不足。因此我们可以根据时间步将序列截断，用前一个序列的final states作为后一个序列的initial states。

![img](https://r2rt.com/static/images/RNN_tf_truncated_backprop.png)

## vanilla RNN

只有一层隐藏层的RNN

# LSTM

用来缓解梯度消失和梯度爆炸。

原先的RNN只有一个内部隐藏状态h，该状态对于短期的输入非常敏感，因此我们增加一个状态——cell state，即c，让它保存长期的状态。

LSTM将隐藏状态和输入拼接在一起，然后乘以一个巨大的矩阵，得到四个门向量。

* i：input gate，whether to write to cell；

$$
i_t = \sigma  (W_i [h_{t-1},x_t] + b_i)
$$

* f：forget gate，how much do we want to forget；

$$
f_t = \sigma (W_f [h_{t-1},x_t] + b_f)
$$

* o：output gate，how much to reveal cell；

$$
o_t = \sigma (W_o [h_{t-1}, x_t] + b_o)
$$

* g：gate gate，how much to write to cell；

$$
g_t = tanh(W_g [h_{t-1}, x_t] + b_g)
$$

cell state的计算公式为：(注意是按元素相乘)
$$
c_t = f_t \circ c_{t-1} + i_t \circ g_t
$$
至于隐藏状态：
$$
h_t = o_t \circ tanh(c_t)
$$
