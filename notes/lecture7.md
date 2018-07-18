# Fancier Optimization



## 随机梯度下降以及各种更新方法



### 普通更新

最简单的形式就是沿着负梯度方向改变参数。比如：

```python
x += - learning_rate * dx
```

其中，学习率learning_rate是一个超参数，固定值。只要学习率足够低，就能在每次变化时使得损失函数降低。

但这种更新会有很多问题，其中一个就是假如loss function在一个梯度方向上很敏感，在另一个方向上则不够敏感，变化很慢。这样就会使得损失函数成之字形往最优点移动。事实上，这个问题在高维空间上会更加显著。

另一个问题就是local minima。SGD会卡在局部最小值——比如局部梯度为0的点。不够事实上，在高维空间，这不算一个问题，这种情况下发送很少，因为不大可能发生所有方向都在向上。

还有一个问题是saddle pointer。在这些位置上，梯度为0或者接近0，但一个方向上梯度向上，另一个方向上梯度下降。这种问题在高维空间比较常见。

![img](https://qph.ec.quoracdn.net/main-qimg-c6faf0305ab9736a3177fe55936b1989)

SGD还有一个问题就是，因为计算损失函数和梯度时N过大，我们可能会采取小批量去计算。但可能会遇到noisy。
$$
L(w) = \frac{1}{N}\sum_{i=1}^{N} L_i(x_i, y_i, W) \\
\nabla_wL(W) =  \frac{1}{N}\sum_{i=1}^{N}\nabla_w L_i(x_i, y_i, W)
$$


### 动量(Momentum)更新

SGD plus momentum：
$$
v_{t+1} = pv_t + \nabla f(x_t) \\
x_{t+1} = x_t - \alpha v_{t+1}
$$

```python
vx = 0
while True:
    dx = compute_grad(x)
    vx = rho * vx + dx
    x += - learning_rate * vx
```

基本思想：保持一个不变的速度，将梯度估计添加到这个速度上，然后在这个速度方向前进。rho是一个摩擦系数，对速度进行衰减，比如可以取0.9。

> 可以想象成一个球，滚下山时，速度在不断变快。有了速度之后，在通过局部极小值点时，因为在这个点仍然有速度（可以理解为惯性），那么即便梯度为0，它也能通过。

另外，原先的SGD可能呈之字形走向最低点，加上动量之后，这些曲折就会抵消。实际上还可以加速在不那么敏感的维度下降。

实际上可以想象成以下的样子，绿色向量是动量，红色是梯度，然而我们实际上步进的是综合权重，即蓝色的向量。

有时我们会看到SGD动量的一些变形，比如，nesterov动量：

![img](http://shuokay.com/content/images/optimization/nesterov.jpeg)

以下就是neterov动量的公式：
$$
v_{t+1} = pv_t - \alpha \nabla f(x_t) \\
x_{t+1} = x_t +v_{t+1}
$$


> 另外，还有一种优化方法——AdaGrad。

```python
grad_squared = 0
while True:
    dx = compute_grad(x)
    grad_squared += dx*dx
    x -= learning_rate * dx / (np.sqrt(grad_squared)+1e-7)
```

这个方法的思想是假如我们有一个很小的梯度，通过累加，相除，加速小梯度的学习速度。同理，能降低大梯度的训练速度。

但这种算法，随着时间的增长，我们的步长在变短，也就是学习率过于激进并且很容易过早停止。

> 还有一种方法：RMSProp

```python
grad_squared = 0
while True:
    dx = compute_grad(x)
    grad_squared = decay_rate * grad_squared + (1-decay_rate)*dx*dx
    x -= learning_rate * dx / (np.sqrt(grad_squared)+1e-7)
```

在这种算法中，我们仍然会累加梯度的平方，但会让该值随着时间主键下降。与adagrad相同，这种算法仍然会在梯度下降慢的维度上加快训练。但这种算法由于dx平方被衰减了，所以学习速率也在变慢。



### Adam

这种算法就是把momentum和AdaGrad/RMSProp结合起来。

```python
first_moment = 0
second_moment = 0
while True:
    dx = compute_grad(x)
    first_moment = beta1 * first_moment + (1-beta1)*dx
    second_moment = beta2 * second_moment + (1-beta2)*dx*dx
    x -= learning_rate * first_moment / (np.sqrt(second_moment)+1e-7)
```

但是这里有一个问题，就是刚开始的时候，哪怕beta2是设置为0.9或者0.99，second_moment仍然很可能非常小，接近于0，这样一开始就会得到一个非常大的步长，而这并不是由梯度引起的。

为了避免这种情况，Adam算法可以加上bias，来避免一开始出现很大的步长。

```python
first_moment = 0
second_moment = 0
for t in range(num_iterations):
    dx = compute_grad(x)
    first_moment = beta1 * first_moment + (1-beta1)*dx
    second_moment = beta2 * second_moment + (1-beta2)*dx*dx
    first_unbias = first_moment / (1-beta1**t)
    second_unbias = second_moment / (1-beta2**t)
    x -= learning_rate * first_unbias / (np.sqrt(second_unbias)+1e-7)
```

> 一般设置beta1为0.9，beta2为0.999，学习率为1e-3

## 学习率

以上所有的方法都用到了学习率这个参数，有时候学习率太大，会导致损失函数爆炸增长；太小了又可能使得训练速度变慢。

有时候我们不一定要用固定的学习率，比如可以使得学习率随着时间衰减。可以考虑比如迭代一万次之后，学习率下降。又可以使得学习率指数下降。一般是基于这样的考虑，如果原来的学习率只会使得损失函数在一个区域里来回变化，那么降低学习率还可以得到更好的效果。

实现学习率退火有三种方法：

* 随步数衰减：没进行几个周期就根据一些因素降低学习率；
* 指数衰减：公式为$\alpha = \alpha_0 e^{-kt}$，其中t是迭代次数，其它是超参数；
* 1/t衰减：公式为$\alpha = \alpha_0 / (1+kt)$；

## Beyond Training Error

事实上，我们不大在意训练误差，我们要考虑的是如何减小训练误差和测试误差之间的gap，准确点说，应该是测试集中更好的结果。

一个方法是模型集成——比起训练一个模型，我们可以从不同的随机初始值训练10个不同的模型。到了测试的时候，我们在10个模型进行测试，然后求10个模型参数的均值。

另一种方法是不用保存多个模型，而是在训练的时候收集多个模型的快照，在测试阶段仍然要求多个模型快照的平均值。（只训练了一次）



# 正则化

Dropout，将激活函数置为零，像是在做集成学习

batch normalization

训练时加入随机性，防止过拟合；测试时应该消除随机性；

数据增强

dropconnect

stocahstic depth

为了防止过拟合，除了集成学习，我们还需要在单一模型里通过控制神经网络的容量来达到这个目的。

* L2正则化：最常见的正则方法，在目标函数中增加一个$\frac{1}{2} \lambda w^2$，对网络中的权重进行惩罚。（之所以加上1/2，是为了求导方便）。使用了L2正则化，就意味着所有的权重都以w += -lambda * W的方向向着０线性下降；
* L1正则化：另一种常见方法，我们增加的是$\lambda w$这一项。

### Dropout

这种正则化方法，是以p的概率将神经元设置为0。故而每个mini batch都在训练着不同的网络，这样就类似于在单个模型中的集成学习。

![img](http://cs231n.github.io/assets/nn2/dropout.jpeg)

```python
""" Vanilla Dropout: Not recommended implementation (see notes below) """

p = 0.5 # probability of keeping a unit active. higher = less dropout

def train_step(X):
  """ X contains the data """
  
  # forward pass for example 3-layer neural network
  H1 = np.maximum(0, np.dot(W1, X) + b1)
  U1 = np.random.rand(*H1.shape) < p # first dropout mask
  H1 *= U1 # drop!
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  U2 = np.random.rand(*H2.shape) < p # second dropout mask
  H2 *= U2 # drop!
  out = np.dot(W3, H2) + b3
  
  # backward pass: compute gradients... (not shown)
  # perform parameter update... (not shown)
  
def predict(X):
  # ensembled forward pass
  H1 = np.maximum(0, np.dot(W1, X) + b1) * p # NOTE: scale the activations
  H2 = np.maximum(0, np.dot(W2, H1) + b2) * p # NOTE: scale the activations
  out = np.dot(W3, H2) + b3
```

> 在测试时，不进行随机失活，但是对于输出需要乘上p。这是因为神经元在训练中输出的时候只有p的概率输出量。但是我们希望测试输出与训练输出有着一致的效果。

如果我们不希望将时间花费在测试时激活数据的计算，也就是避免测试性能的降低，我们可以进行方向的随机失活：

```python
""" 
Inverted Dropout: Recommended implementation example.
We drop and scale at train time and don't do anything at test time.
"""

p = 0.5 # probability of keeping a unit active. higher = less dropout

def train_step(X):
  # forward pass for example 3-layer neural network
  H1 = np.maximum(0, np.dot(W1, X) + b1)
  U1 = (np.random.rand(*H1.shape) < p) / p # first dropout mask. Notice /p!
  H1 *= U1 # drop!
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  U2 = (np.random.rand(*H2.shape) < p) / p # second dropout mask. Notice /p!
  H2 *= U2 # drop!
  out = np.dot(W3, H2) + b3
  
  # backward pass: compute gradients... (not shown)
  # perform parameter update... (not shown)
  
def predict(X):
  # ensembled forward pass
  H1 = np.maximum(0, np.dot(W1, X) + b1) # no scaling necessary
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  out = np.dot(W3, H2) + b3
```



# 迁移学习

一个功能强大的网络在面对小数据的时候很可能出现过拟合的问题，一个方法是使用正则化，另一个方法就是迁移学习。

迁移学习，顾名思义就是将已经训练好的模型参数迁移到新的模型来帮助新模型的训练。

打个比方，你可以现在一个大数据集上进行训练。然后现在你有了新的想法，希望从原来的大数据集训练出来的提取特征的能力能应用到自己感兴趣的小数据集上。那么一般的做法就是修改从最后一层的特征到最后的分类输出之间的全连接层，重新初始化这部分矩阵，冻结前面的权重。现在就只需要训练一个新的线性分类器，训练最后一层，让它在我现在的数据集上收敛。

在做迁移学习时，有这样的一个场景：

|          |      与源数据集相似      | 与源数据集不相似 |
| :------: | :----------------------: | :--------------: |
| 小数据集 | 在顶层使用一个线性分类器 | 重新初始化大网络 |
| 大数据集 |         微调模型         |  精调大部分网络  |

