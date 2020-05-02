# 【论文精讲】 - Word2Vec

> [Mikolov, Tomas, et al. "Distributed representations of words and phrases and their compositionality." Advances in neural information processing systems. 2013.](http://papers.nips.cc/paper/5021-distributed-representations-of-words-andphrases)
>
>思路：论文章节为主线，并结合其它优秀论文添加相关知识点。

- ## 目录

  - [概要](#%e6%a6%82%e8%a6%81)
  - [介绍](#%e4%bb%8b%e7%bb%8d)
  - [Skip-Gram 模型](#skip-gram-%e6%a8%a1%e5%9e%8b)
  - [实验结果](#%e5%ae%9e%e9%aa%8c%e7%bb%93%e6%9e%9c)
  - [组合单词](#%e7%bb%84%e5%90%88%e5%8d%95%e8%af%8d)
  - [总结](#%e6%80%bb%e7%bb%93)
  - [参考论文](#%e5%8f%82%e8%80%83%e8%ae%ba%e6%96%87)

## 概要

在该论文发表之前，`skip-gram` 模型大火，能够**有效**从无结构语料库中学习到语法和语义信息，此论文基于`Skip-Gramm` 模型提出一些以提升词向量训练效果和提升训练速度的方法(`SubSampling`,`Negative Sampling`)。对高频词的`下采样`方法能够有效提升训练速度和语法的学习。

**问题的提出：** 目前（当时），词向量表示有一个很重要的缺陷：无法表示词序的不同性和口语化词组的表示。

什么意思呢？比如这里有一个两个单词："Canada","Air"，可是一旦遇到"Air Canada"（加拿大航空公司）这个单词，就无法表示其意思。在日常用语中有很多类似由两个单词进行拼接而得的词组，比如"Boston River"等。由此作者开始尝试对此类**组合词组**使用词向量表示的工作，方法很简单，效果很好，就是这么暴力。

## 介绍

不像之前的全连接神经网络，此论文中扩展后的`Skip-Gram`模型由于没有使用全连接矩阵连接，能够大幅度减少权重数量，从而加快训练的速度。

无论是全连接神经网络模型还是skip-gram模型都能够从语料库中学习到一些**模式**，而这些模式就是学习得来的语法和语义信息，甚至直接可以使用词向量进行线性表示，那么

**词向量如何表示一定的语法和语义信息呢？**

我想大家肯定见过这样的例子：$Vec(played) - Vec(play)$ 和 $ Vec(worded) - Vec(work)$ 很相似，这个一般为语法信息。而 $Vec(Madrid) - Vec(Spain)$ 和 $Vec(France) + Vec(Paris)$ 很相似，这个一般为语义信息。

由此可以看出，词向量可以有效学习语料库中的不同层次的模式，并将不同的信息在词向量的不同维度上表现出来，毕竟，词向量的每一维度都可以看成一个特征。

上面说过，扩展后的`skip-gram`可以有效提升训练速度，如果大伙儿如果连基础的`skip-gram`都不了解，下面的内容就没办法讲下去了。来，看看`skip-gram`是如何逼我们必须要对其进行扩展的。



## Skip-Gram 模型
## 实验结果
## 组合单词
## 总结

## 参考论文：
- [](http://papers.nips.cc/paper/5021-distributed-representations-of-words-andphrases)
- [Efficient estimation of word representations in vector space](https://arxiv.org/abs/1301.3781)
