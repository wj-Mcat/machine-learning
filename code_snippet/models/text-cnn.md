# 论文

[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

> 以下代码提供参考，如果需要使用时，可直接复制粘贴即可



# pytorch-text-cnn

## version 1

```python
# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextCNN'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)


'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
			# embedding 可以微调
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

```

# tensorflow-text-cnn

```python

from hbconfig import Config
import tensorflow as tf



class Graph:

    def __init__(self, mode, dtype=tf.float32):
        self.mode = mode
        self.dtype = dtype

    def build(self, input_data):
        embedding_input = self.build_embed(input_data)
        conv_output = self.build_conv_layers(embedding_input)
        return self.build_fully_connected_layers(conv_output)

    def build_embed(self, input_data):
        with tf.variable_scope("embeddings", dtype=self.dtype) as scope:
            embed_type = Config.model.embed_type

            if embed_type == "rand":
                embedding = tf.get_variable(
                        "embedding-rand",
                        [Config.data.vocab_size, Config.model.embed_dim],
                        self.dtype)
            elif embed_type == "static":
                raise NotImplementedError("CNN-static not implemented yet.")
            elif embed_type == "non-static":
                raise NotImplementedError("CNN-non-static not implemented yet.")
            elif embed_type == "multichannel":
                raise NotImplementedError("CNN-multichannel not implemented yet.")
            else:
                raise ValueError(f"Unknown embed_type {self.embed_type}")

            return tf.expand_dims(tf.nn.embedding_lookup(embedding, input_data), -1)

    def build_conv_layers(self, embedding_input):
        with tf.variable_scope("convolutions", dtype=self.dtype) as scope:
            pooled_outputs = self._build_conv_maxpool(embedding_input)

            num_total_filters = Config.model.num_filters * len(Config.model.filter_sizes)
            concat_pooled = tf.concat(pooled_outputs, 3)
            flat_pooled = tf.reshape(concat_pooled, [-1, num_total_filters])

            if self.mode == tf.estimator.ModeKeys.TRAIN:
                h_dropout = tf.layers.dropout(flat_pooled, Config.model.dropout)
            else:
                h_dropout = tf.layers.dropout(flat_pooled, 0)
            return h_dropout

    def _build_conv_maxpool(self, embedding_input):
        pooled_outputs = []
        for filter_size in Config.model.filter_sizes:
            with tf.variable_scope(f"conv-maxpool-{filter_size}-filter"):
                conv = tf.layers.conv2d(
                        embedding_input,
                        Config.model.num_filters,
                        (filter_size, Config.model.embed_dim),
                        activation=tf.nn.relu)

                pool = tf.layers.max_pooling2d(
                        conv,
                        (Config.data.max_seq_length - filter_size + 1, 1),
                        (1, 1))

                pooled_outputs.append(pool)
        return pooled_outputs

    def build_fully_connected_layers(self, conv_output):
        with tf.variable_scope("fully-connected", dtype=self.dtype) as scope:
            return tf.layers.dense(
                    conv_output,
                    Config.data.num_classes,
                    kernel_initializer=tf.contrib.layers.xavier_initializer())
```







