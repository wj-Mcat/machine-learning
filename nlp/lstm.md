# 闲聊
🤔 人们能够理解语言，从来不是从**零**开始，而是根据**以往经验**来做出判断。故“以往经验”对于判断与认知是十分重要的，设想一下，人类如果没有“经验”（记忆）或只有三秒记忆，那跟咸鱼有什么区别，人类就不会有什么文明。
LSTM 就能让**机器**开始记住他们所读过的内容。

***********

# RNN
## 概念
RNN能够在一定程度上记住之前的知识，通过训练序列数据，能够记住某词前面出现的词语，这样就能产生些许“**经验**”。不要以为这个功能很简单，其实大有作用，就好比如，你说上一句，它就知道下一句是啥？
而且实践证明RNN在经验方面已经开始崭露头角🙉，不过仍需进一步研究。 要知道其工作原理，需要了解去模型图。如下图所示
![LSTM总概图](https://img-blog.csdnimg.cn/20190912101449360.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8yOTYyMzU1NQ==,size_16,color_FFFFFF,t_70#pic_center =130x)

## 公式图
中某一个单元，Ht为输出序列的某一个单元。上图是RNN缩略图，因是一个自循环图，看起来有些困难，将其展开后如下图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190912101912807.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8yOTYyMzU1NQ==,size_16,color_FFFFFF,t_70)

## 栗子

举一个非常简单的例子，让RNN看10000次"**我爱你，我也爱你 💖💖💖**" 后，此时她的眼里只有你，当我们问RNN："我爱你"时，RNN会毫不犹豫的回答"我也爱你💖💖💖"。此案例中，RNN眼里只有[我,爱,你,也]四个词，甚至都没有一个**不**字，并且在她的认知里面，“我爱你”后面只会有“我也爱你”的情况，没有第三+者，这只是一个很简单的例子，可现实中的情况会复杂的多，RNN会看各种各样的事情，出现的频率也不同，则根据前后序列的关联性创建不同的权重。
> 好了，再举一个栗子，你的那个她一生中会经历很多不同的异性，出现的频率都不同，可是如果你想要让她的眼里只有你的话，就需要经常在她的**眼前转悠**，慢慢让她熟悉你，**眼里只有你**，时间久了，就会离不开你，哈哈哈~~~这个经验对于很多方面都还是比较有效的。
> 没想到在RNN中还学到了一招儿~~~

**********



# LSTM术语
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190912174614962.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8yOTYyMzU1NQ==,size_16,color_FFFFFF,t_70)
🔥 上图就是LSTM内部的模型图。在介绍LSTM之前，cue一下基本符号与概念：

- 基本符号
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190912174658384.png)
- Cell State
  存储当前Cell的状态信息，包含数据的序列信息。比如当输入为“I love you”时，第三个cell中能够保存“I”和“Love”的信息，那么第三个Cell就可以大概率预测出“you”。
- Gate
	Gate主要是控制在Cell State添加或删除信息，由上图中的Sigmoid函数和点乘操作组成。每个输入和历史CellState都是需要进行选择性记忆和遗忘，这个记忆和遗忘就是在不同阶段给不同对象不同的权重。比如LSTM在读过“I Love”之后，“you”的记忆可能比较明确和清晰（权重比较大），而“him”的记忆可能比较模糊（权重比较小），究其原因还是因为“you”在“I Love”附近转悠的次数比较多而已。
此外，Sigmoid函数输出在(0,1)之间，当输出为0的时候，不让当前信息流入到Cell State中；当输出为1的时候，就当前信息全部流入到Cell State中。
- $_{x_t}$是输入序列中第t个元素
- $_{h_t}$是输出序列中第t个元素

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190912175622897.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8yOTYyMzU1NQ==,size_16,color_FFFFFF,t_70#pic_center)
上图展示了$_{C_t-1}$如何变成$_{C_t}$。LSTM中的关键是**每个单元格的状态**（cell state），而这个单元格状态就类似于**传送带**，能够将一些信息沿着序列数据一直传递到序列**最低端**，且每经过一个cell，都能够携带上该单元格的信息，这样后面的单元格是知道**之前**序列的数据，故具有**记忆**的功能。

************

# 细说LSTM
接下来将一步一步的讲解LSTM中不同**模块**，逐层解剖出核心含义。

（1）***选择性遗忘***

首先第一步就是要**决定之前的经验($_{C_t-1}$)，有多少能够流向下一个Cell State($_{C_t}$)**，具体公式如下图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190912180252140.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8yOTYyMzU1NQ==,size_16,color_FFFFFF,t_70)

如何遗忘，是由一个“**Forget Gate Layer**”控制的，上图所示便是。由$_{h_t-1}$和$_{x_t}$经过$sigmoid$函数之后输出一个(0,1)的数字，而这个数字就决定了上一个Cell State 有多少信息能够**流向**下一个Cell State。

当LSTM看了很多“我喜欢打篮球”的语句时，突然来了十句“我喜欢打乒乓球”，此时就需要将乒乓球**加入**到我的爱好列表中，从另一个层面看，需要对“打篮球”进行一定程度的**遗忘**，才能空出来“打乒乓球”。因为这是有权重比例的，而所有的爱好比例总和为1（**非严格意义上来讲**）。而每个爱好可能在不同上下文环境中的比例是不一样的。

（2）***选择性记忆***
接下来就是要根据当前序列元素的输入和上一个输出来选择性记忆，决定哪些信息需要存储到Cell State中。而这个过程就包含两个过程，如下图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190912180624989.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8yOTYyMzU1NQ==,size_16,color_FFFFFF,t_70)

第一个过程是由“**Input Gate Layer**”组成，此层决定了我们要把输入信息中的哪些信息输入嵌入到当前的Cell State中。第二个过程就是通过Tanh创建一个$\tilde{C_t}$通过与 $_{i_t}$ 的组合来决定哪些信息是需要保存下来的。

在实际案例中就类似于上文中的“我喜欢打乒乓球”选择性的**加入**到到新的Cell State中。$_{i_t}$ 和 $\tilde{C_t}$ 共同决定“乒乓球”加入到我的爱好列表中。

（3）***New Cell State***
Old Cell State是如何变成New Cell State呢？如下图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190912181012894.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8yOTYyMzU1NQ==,size_16,color_FFFFFF,t_70)

首先对Old Cell State进行选择性遗忘，然后对Input进行选择性记忆，然后加起来就是New Cell State。这个过程就比较容易理解了。

（4）***输出***
输出是根据当前Cell State，上一个输出，和当前输入决定的。具体的逻辑图如下所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190912181051430.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8yOTYyMzU1NQ==,size_16,color_FFFFFF,t_70)

与“**Forget Gate Layer**”相反， **通过上一个输出和当前输入，决定当前输入信息有多少信息可以影响输出**。最终生成一个(0,1)的比例数据。另外上一个 Cell State 通过tanh后再和 ${o_t}$ 点乘， 就可生成当前Cell 的输出 ${h_t}$。


LSTM是具有在cell上**添加**或**删除**信息的功能，而这个功能结构化的表现为Gate，中文理解为门阀。**门阀**是让数据能够一直在序列数据上流下去的关键。上图就是一个门阀，由一个**sigmoid函数**和矩阵**点乘**组成。

在实际应用当中，有这样的一个场景，我们要判断一个动词是单数还是复数，就需要知道主语是单数还是复数。


# summary

🔥 LSTM能够**保存**训练过程中的**序列**信息，并通过**Gate**能够决定哪些信息需要记住，删除，和读取，就类似于计算机的内存存储。每个Cell能够根据对应输入和上一个输出决定接下来我要存储哪些信息，遗忘哪些信息，然后组装成记忆存储起来，同时产生对应的输出。

# 梯度消失
???  🙌 下一次再讲，**see you later** ~~~~~~~~


参考链接：
- [Understanding LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [wiki-lstm](https://skymind.ai/wiki/lstm)
- 