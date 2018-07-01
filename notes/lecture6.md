# 卷积神经网络

卷积层可以保存空间结构，全连接层要展开

将卷积核与图片空间结构重叠，相应的元素进行相乘

采用一组卷积核，就能得到一组激活映射

前面几层可能代表一些低阶的特征，比如边沿之类的，后面代表高阶的特征

卷积核似乎在图片寻找某些特征

卷积核在滑过图片时，可以自定义步长，这样产生的输出尺寸也会不一样

输出尺寸 = (N-F)/stride+1，没有填充的前提下

可以填充：zero pad



池化层

一般不作深度的降采样

max pooling：只取最大值

池化层一般不作padding



后面层代表的是一组复合模板的受激程度

---------------------------------------------------------------------------------------------------------

## 概述

与一般的神经网络不同，传统的神经网络采用全连接的方式，在面对大尺寸图像时效果不佳。打个比方，对于一张200x200x3的图像，网络中的神经元将会包含120000个权重值，而神经元又不止一个，这样将会导致参数的数量快速增加，使得效率低下。

而卷积神经网络的神经元并不是采取全连接的方式，而是每个神经元仅仅与前一层中的一小块区域连接。以下图为例：

第一幅图是传统的神经网络：

![img](http://cs231n.github.io/assets/nn1/neural_net2.jpeg)

第二幅图是卷积神经网络：

![img](http://cs231n.github.io/assets/cnn/cnn.jpeg)



## 构建卷积网络的层类型

一个卷积神经网络主要是有三个类型的层：卷积层、池化层(pooling)和全连接层。

在卷积层中，每个神经元与输入层中的一小块区域相连，然后计算该空间区域与自己图像权重的内积。如果我们使用n个核，那么输出的数据维度的深度也为n。

RELU层没有参数，它会逐个元素使用激活函数进行操作，在该层中，数据尺寸不发生改变。

池化层主要是进行降采样的操作，修改空间维度，一般只修改宽度和高度，不修改深度。

全连接层与在传统神经网络中的一样，计算分类评分。

![img](http://cs231n.github.io/assets/cnn/convnet.jpeg)



### 卷积层

卷积层是构建卷积神经网络的核心层，是由一些可学习的滤波器(filter)组成的。每个filter在空间上（高度与宽度）都比较小，深度与输入数据一致。在前向传播的时候，filter在输入数据的宽度和高度上滑动（卷积），生成一个2维的激活图。该层网络会让滤波器学习到当它看到某个类型的特征就会被激活。



#### 局部连接

对于高维度的图片处理来说，神经元进行全连接是不现实的，相反，我们让每个神经元只与输入数据的一小块区域连接。该区域就叫做神经元的感受域(receptive filed)。它的尺寸就是filter的尺寸。

> 如果输入数据体的尺寸为[16x16x20]，感受域的尺寸是3x3，那么卷积层的每个神经都会和输入数据体有3x3x20个连接，也有3x3x20个权重



#### 空间排列

控制输出数据的尺寸上，有三个超参数：深度、步长和零填充。

* 深度：输出数据体的深度与filter的数量一致。沿着深度排列，感受域相同的神经元集合可能会被不同方向的边界或者是颜色斑点所激活；
* 步长：在滑动滤波器的时候，必须指定步长，步长对于宽和高是等效的；
* 零填充：可以使得输入数据尺寸与输出相同；

> 计算公式：输入尺寸(W)，感受域尺寸(F)，步长(S)，零填充数量(P)，那么输出空间尺寸为：**(W-F+2P)/S+1**

超参数之间是会有限制的，即需要保证计算得到的输出空间尺寸为整数。



#### 参数共享

参数共享是用来控制参数个数的，举一个例子，如果输出数据有n个神经元，每个神经元有m个参数，那么合起来的参数个数将会非常巨大。而且单一层的参数就那么多，总共加起来就更难说了。

那么实际上，深度列上的每个神经元都是与输入数据体的同一个区域连接的，当然权重不同。但如果将深度维度上一个单独的2维切片看作是深度切片，在参数共享的情况下，一个卷积层就只有d(代表深度)个权重集，因为每个深度切片用的权重是相同的。



### 池化层

通常，在连续的卷积层之间会周期性地插入一个池化层，用来降低数据体空间的尺寸（主要是宽度和高度），这样降采样能有效地减少网络中参数的数量，减少消耗资源，控制过拟合。

一种常见的池化操作就是max操作，对输入数据体的每一个深度切片进行独立操作，例如可以从4个数中选取最大值，深度不变。
