# 深度学习框架



## CPUvsGPU

cpu就是一个中央处理器，而GPU图形处理单元。GPU拥有更多的核，但每个核运行速度非常慢，并且能处理的指令比较少，GPU的核也是不能独立工作的。至于内存，CPU是与系统共享内存，GPU则是内置有内存。

一个很典型的可以使用GPU的算法就是矩阵相乘，能实现高并行。

## DeapLearning 框架

一个大的系统，其计算图非常复杂。而深度学习框架有以下优点：

* 建立计算图
* 计算梯度
* 忽略硬件等影响，方便在GPU运行

> numpy只能在CPU上运算

### pytorch

Pytorch是一个基于Python的科学计算包：

* 用GPU运行的numpy
* 灵活度高和速度快的深度学习平台

#### Tensors

Pytorch的Tensors类似于Numpy的ndarrays，另外tensors可以被用于GPU的并行计算。

构造一个5x3的，没有被初始化的矩阵：

```python
x = torch.empty(5, 3)
print(x)
'''
tensor([[                                 0.0000,
                                          0.0000,
               -654847608700481908254965760.0000],
        [                                 0.0000,
         -405710708045017320182827431493632.0000,
                                          0.0000],
        [                                 0.0000,
                                          0.0000,
                                          0.0000],
        [                                 0.0000,
         -453844602137521297376523904352256.0000,
                                          0.0000],
        [-439010366646764585542407698776064.0000,
                                          0.0000,
               -665030211429169580746997760.0000]])
'''
```

构造一个随机的初始化矩阵：

```python
x = torch.rand(5, 3)
print(x)
'''
tensor([[0.4482, 0.0196, 0.1131],
        [0.4095, 0.2553, 0.5499],
        [0.3263, 0.3233, 0.8145],
        [0.0429, 0.1034, 0.5939],
        [0.8096, 0.0853, 0.3248]])
'''
```

构造一个指定类型的zero矩阵：

```python
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
'''
tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])
'''
```

也可以直接从一个list构造tensor

```python
x = torch.tensor([5.5, 3])
print(x)
'''
tensor([5.5000, 3.0000])
'''
```

另外，也可以使用一个tensor去初始化新的tensor，新的tensor会继承老的tensor的属性，除非你重写这些属性：

```python
x = x.new_ones(5, 3, dtype=torch.double)
print(x)
x = torch.randn_like(x, dtype=torch.float64)
print(x)
print(x.size())
'''

tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
tensor([[-1.7981, -2.5819, -0.8203],
        [-0.8029, -0.2673,  2.6177],
        [-0.4584, -1.1823, -0.9143],
        [-0.1608, -0.3784, -0.8696],
        [-1.0211,  0.5049, -0.5876]], dtype=torch.float64)
torch.Size([5, 3])
'''
```

#### Operations

以加法为例，pytorch提供了几种加法操作：

```python
y = torch.rand(5, 3, dtype=torch.double)
print(x + y)

print(torch.add(x, y))

result = torch.empty(5, 3, dtype=torch.double)
torch.add(x, y, out=result)
print(result)

y.add_(x) # y = x + y
print(y)

'''
tensor([[-1.5982, -2.3207, -0.0049],
        [ 0.0473, -0.1795,  3.1768],
        [-0.2709, -0.4187, -0.3739],
        [-0.0311, -0.1798, -0.3391],
        [-0.5565,  0.5051, -0.4602]], dtype=torch.float64)
'''
```

> 一些带有后缀 **_** 的操作符，都能改变使用操作符的变量。例如x.copy_(y)，x.t\_()将会修改x

你也可以使用标准的numpy的操作来进行操作：

```python
print(x[:, 1])
'''
tensor([-2.5819, -0.2673, -1.1823, -0.3784,  0.5049], dtype=torch.float64)
'''
```

而对于resize，可以使用torch.view对tensor进行重新构造：

```python
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8) # -1 可以进行自动推导维度
print(x.size(), y.size(), z.size())
'''
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
'''
```

如果是只有一个元素的tensor，可以使用**.item()**取出一个值：

```python
x = torch.randn(1)
print(x)
print(x.item())
'''
tensor([0.1186])
0.11860071122646332
'''
```

#### Numpy Bridge

Torch的tensor可以转成Numpy array，此时它们共享底层的内存(numpy也可以转换成tensor)：

```python
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

'''
tensor([1., 1., 1., 1., 1.])
[1. 1. 1. 1. 1.]
tensor([2., 2., 2., 2., 2.])
[2. 2. 2. 2. 2.]
'''
```

#### CUDA

Tensor能在不同的设备之间进行转移

```python
if torch.cuda.is_available():
    device = torch.device('cuda')
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x+y
    print(z)
    print(z.to("cpu", torch.double))
'''
tensor([1.1186], device='cuda:0')
tensor([1.1186], dtype=torch.float64)
'''
```





