# 深度学习中的优化算法

## 一、概览

优化算法的基础在于梯度下降，故需要深入理解其原理过程，才能够掌握优化的目标和过程。

李宏毅老师和吴恩达老师深入浅出的介绍了梯度下降的不同层面的信息，看他们的视频是最好的学习资料，在此我给出了B站上面搬运的视频地址：

- [吴恩达](https://www.bilibili.com/video/BV164411b7dx?p=9)
- [李宏毅](https://www.bilibili.com/video/BV13x411v7US?p=6)

## 二、Batch Gradient Descent

在一组数据集上只需要更新一次梯度，而非是每个数据计算损失后都要计算一次梯度。

- 公式
  $$
  \theta_{i+1}=\theta_{t}-\eta g_{t}
  $$

- 优点

  - 减少个别数据对整体梯度上的影响。

- 缺点

  - 由于是在一组数据集上计算出总体损失后再来更新梯度，故总体计算速度非常慢。

- 伪代码

```python
for i in range ( nb_epochs ):
    params_grad = evaluate_gradient ( loss_function , data , params )
    params = params - learning_rate * params_grad
```



## 二、SGD

### 2.1 公式

$$
\theta_{i+1}=\theta_{t}-\eta g_{t}
$$

### 2.2 优点

步骤是最简单的

### 2.3 缺点

- 没有动量概念
- 收敛速度慢



参考论文：

- [An overview of gradient descent optimization algorithms](http://arxiv.org/abs/1609.04747)
- [Gradient Descent based Optimization Algorithms for Deep Learning Models Training](http://arxiv.org/abs/1903.03614)

