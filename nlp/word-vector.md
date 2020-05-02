# 词向量的魅力

> 为了将这个问题说清楚，参考了很多资料，最终采用以下五篇论文的研究为结论给大家细细描述，希望能够将这个问题说清楚

# 目录

- 介绍
- Word-Vector 是如何携带信息的



# 介绍

查阅了五篇论文，

词向量是一个多维带有权重的向量。

如何理解词向量分布式表示：这个问题应该用`one-hot encoding`做引子。`one-hot encoding`中词库大小等于词向量维度的大小，比如词库大小为`10000`维，每个单词用某一维的`1`来携带信息，其他维度都是0，这种表达方式是最原始，词与词之间毫无语义关联的一种表达方式。无论是从人类还是从机器的角度，很难进行学习。那么`Word2Vec`就是才好用`分布式表示`( *distributed* representation )，翻译过来很容易给大家造成误解，这里的分布式和分布式网络有啥关系，其实原理是一样的：

**从某个角度将，分布式网络能够将网络访问的压力负载均衡到每个机器上，而词向量是将语义的表达从某一个维度的表达分散到每一维度，每个维度都是某个特征的权重，至于是什么特征，人无法解释，可是机器能够知道。**

马克思对人的本质定义：**人是所有社会关系的总和**。

**这些词以一种非常抽象的方式来表达语义，每一维代表着一个该特征的权重，所有权重代表着这个词的含义**

## word vector 是如何携带信息的

`wordvector`在`NLP`领域是基础工作，里面携带的信息量决定着整个模型能够学习到的知识，那么WordVector是如何携带信息的呢？

众所周知，一个词的词向量是一个一维向量，维度可以自己设置。举个栗子：

![TextCNN](D:/School/BUPT/first_year/projects/maching-learning-and-cv/assert/imgs/animal-word-vector.png)

上图中词向量维度是4，每个维度代表着不同的特征，有：

- `animal`：是动物的指数
- `domesticated`：是否被驯化的指数
- `pet`：是宠物的指数
- `fluffy`：是毛茸茸的指数

以上四个指数代表着一个词向量中四个维度的特征指标，而每个词在每个特征的指标不同。如果对数据挖掘比较熟悉的同学就知道，这就已经类似于一个泰坦尼克号生存者分析的问题了。只是这里的数据全部是连续的。

**回到真实的词向量上面来，每个词向量都可能有200维左右，每一维都代表着一个特征。当使用欧氏距离计算词的相似度时，当两个词相似度很高，说明两者的欧氏距离比较相近，进一步说明两个词每个维度的值都非常相近。反过来也可以说得通。**

> 词向量将相似度映射到多维空间中的临近点。

到此，大家就应该知道了，词向量是如何携带信息的。

**反观，传统的`one-hot encoding`和`bag-of-words`都无法捕捉到细粒度的特征相关性，无法捕捉到语法（结构）和语义（含义）的关系，因为他们本身就是使用一种毫无关联性的方式来表达，本身携带的信息就非常小。




## 词向量是如何携带语义信息的

![TextCNN](D:/School/BUPT/first_year/projects/maching-learning-and-cv/assert/imgs/man_woman_small.jpg)

这张图相信不陌生吧，这是[`glove`]( https://nlp.stanford.edu/projects/glove/ )对于词的描述图。因为词向量赋予词计算能力，词向量：

$$Vector(woman) - Vector(man) = Vector(queen) - Vector(king)$$

写到这里，我忍不住想要变下形：

$$Vector(woman) =  Vector(man) + Vector(queen) - Vector(king)$$

有木有感觉到很神奇，其实相似的词对之间，不同维度的特征值差也是相似的，可能是一个维度，也有可能是多个维度的组合，总之是能够通过不同维度的**特征指标平移**从而得到另外一个词对。

**相似性**

那么两个词的相似性归根结底还是从语料中学习到的：`man`和`woman`通常是由相似的上下文环境，比如`man`在一个句子中出现过，`woman`也会在类似的句子中出现过，甚至很有可能整个句子的区别就是`man`单词和`woman`单词的区别。此时我们就可以说，`man`和`woman`是相似的。

## 为什么要使用WordVector作为自然语言的输入

通过语料，可以用一种简单的方式就可以学习到语法和语义的信息，并且这种信息能够在向量的简单运算中就可以体现，比如名词单数转复数在向量中的体现如下所示：

 ***xapple – xapples ≈ xcar – xcars, xfamily – xfamilies ≈ xcar – xcars*** 

根据以上公式我们可以知道，词向量可以本能的回答这个问题：**A对于B就像C对于？**我们通过简单的向量运算就可以知道结果，然后在词向量空间中**找到**出最相似的那个词就结果了。



## 词向量的生成

感受到词向量的强大之后，那就会很好奇，这个词向量是如何生成的呢？一般使用以下两种方法：

- 词频统计/词频上下文
- 预测上下文（比如[word2vec]( https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf )中就包含skip-gram 和 bag-of-words）

过程太复杂，详细请看[word2vec]( https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf )和[glove](https://nlp.stanford.edu/pubs/glove.pdf)论文，或者国外大佬写的这几个博客：

- 

- 其实word2vec是一个监督学习任务
- 



## 实现的工具

现在常用的方法有以下几种：

- [Word2Vec]( https://radimrehurek.com/gensim/models/word2vec.html )
- Glove
- Deeplearning4j



## 参考链接

- [Word2Vec]( https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa )
- [the-amazing-power-of-word-vectors]( https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/ )
-  [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781.pdf) – Mikolov et al. 2013 
-  [Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) – Mikolov et al. 2013 
-  [Linguistic Regularities in Continuous Space Word Representations](http://www.aclweb.org/anthology/N13-1090) – Mikolov et al. 2013 
-  [word2vec Parameter Learning Explained](http://arxiv.org/pdf/1411.2738v3.pdf) – Rong 2014 
-  [word2vec Explained: Deriving Mikolov et al’s Negative Sampling Word-Embedding Method](http://arxiv.org/pdf/1402.3722v1.pdf) – Goldberg and Levy 2014 