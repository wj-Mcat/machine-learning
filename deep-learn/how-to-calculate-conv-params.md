## ·

### 一、背景
这段时间在用Keras搭建卷积神经网络中，发现Keras可以自行计算出每层中对应参数的数量，对计算过程我比较好奇：

### 二、问题以及解决方案

### 2.1 问题环境描述

有一张224*224大小的图片，第一层就为卷基层，代码如下：

```python
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()
model.add(Conv2D(16,(2,2),input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dense(133))
model.summary()
```

最终统计结果如下：

![](./imgs/neural-layer-params.png)


> 这里先抛出卷积层中卷积核操作动态图：

![](./imgs/neural-cell-2.gif)

### 2.2 第一层（卷积层）的参数个数计算

先整理以下此环境中对应的数据信息

- 信息列表
    - Filter个数：32
    - 原始图像shape：`224*224*3`
    - 卷积核大小为：`2*2`
- 一个卷积核的参数：
    `2*2*3=12`
- `16`个卷积核的参数总额：
    `16*12 + 16 =192 + 16 = 208`
    
    > `weight * x + bias`根据这个公式，即可算的最终的参数总额为：`208`


### 2.3 第二层（池化层）的参数个数计算

很明显，没有嘛。不清楚的，去看看池化层到底是[啥玩意儿](https://www.cnblogs.com/zf-blog/p/6075286.html)！    

### 2.4 第三层（全连接层）的参数个数计算
s
第二层池化层的输出为：`(111*111*16)`，从第一层到第二层，只是图片大小发生了变化，深度没有发生变化，而``Dense``对应的神经元个数为`133`个，那么还是根据公式：`weight * x + bias`，计算得：`133*16+133=2261`


# 三、总结

在卷积层中，每个卷积核都能够学习到一个特征，那么这个特征其实就是卷积核对应学习到的参数：矩阵中每个单元weight，以及整个卷积核对应的一个bias

> 解决这个问题最根本还是需要理解最本质的一个公式：`weight * x + bias`