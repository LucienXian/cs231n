# 图像分类



## 数据驱动方法

图像识别时，计算机接收图像，并从固定的标签从选择一个标签，计算机看到的图像其实是一个多维数组。

但改变相机角度，环境光度，图像变形，遮挡，与背景重叠，类内差异等情况，计算机的识别算法应该是robust的。

实际上，我们不会考虑为某个是别的物体硬编码某个规则，而是收集图像数据和标签，然后用机器学习去训练分类器，再利用分类器去预测标签。这就是一种数据驱动的算法。

在比较图像时，我们可以使用L1 distance: $d_1(I_1, I_2)=\sum_{p}|I_1^2-I_2^2|$

对于NNC，最近邻分类器，它的train是O(1)，而predict是O(N)，而我们希望的是训练可以慢一点，而测试要快一点。另外就是，最近邻可能会使得分类不够泛化，因此我们使用knn来解决这个问题。

下面这张图很好地表示了knn带来的效果，使得分类更加平滑：

![img](https://pic4.zhimg.com/80/51aef845faa10195e33bdd4657592f86_hd.jpg)

## k最近邻算法

L1距离有坐标依赖，如果特征有特别的意义，一般选择L1距离。但最好都尝试一下。

L2距离

k和distance：hyperrparameters

超参数的设置：

* 将参数分为三组：train、validation、test，validation选择参数。
* cross-validation：将数据分成多组，轮流将每组选做validation，然后平均其结果。

knn很少在图像上使用，一方面是太慢了，另一方面则是距离向量的计算不适合图像像素

## 线性分类I

相对于knn，线性分类需要参数。

image->f(x, W)=Wx+b->10 number giving class scores

打个比方，以下的图可以看作

![img](https://pic4.zhimg.com/80/7c204cd1010c0af1e7b50000bfff1d8e_hd.jpg)

对于图像来说，我们可以将图像想象成高维空间的一个点，线性分类器就是一条分割线。

## 问题

1. 搭建环境时遇到了这个问题

   > Unable to compile python Pillow

也就是无法编译pillow，导致在virtualenv中无法pip要求的库。从这里找到了答案：https://github.com/Microsoft/WSL/issues/256，把依赖装上

**sudo apt-get install libtiff5-dev libjpeg8-dev zlib1g-dev libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python-tk**

2. 用脚本，下载数据集太慢了，还不如手动下载
3. pip install -r requirements.txt 时遇到了site无法安装，按照[这里](https://stackoverflow.com/questions/48696858/unable-to-install-distribution-site)的说法，site是内置的。

> [`site`](https://docs.python.org/3/library/site.html) is internal to the Python interpreter, and is used to initialize machine-specific details of your Python installation.
>
> What's telling you that you need to install this module? Ignore it. It's wrong.