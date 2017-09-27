# -*- coding:utf-8 -*-
"""
@author: gxjun
@file: model.py
@time: 17-9-26 上午11:52
"""
import tensorflow as tf
import numpy as np
from config import cfg

# type of the model
funcMap = {'rnn': tf.contrib.rnn.BasicRNNCell,
           'gru': tf.contrib.rnn.GRUCell,
           'lstm': tf.contrib.rnn.BasicLSTMCell
           }


def model(X, label, vocab_size, num_layers=2, batch_size=64,
          learning_rate=0.01):
    """
    训练模块
    :param X: 训练数据
    :param label: 标签数据
    :param vocab_size:  one_hot维度
    :param rnn_size: W权重个一行的个数（也可以成为神经元的个数）
    :param num_layers: 网络的层次
    :param batch_size: 批次大小
    :param learning_rate: 学习率
    :return:
    """

    graph = {}
    if funcMap.get(cfg.model, None) is None:
        print ("type of the model is not existing ! you choose model type: {}".format(cfg.model))
        return None

    cell = funcMap[cfg.model](cfg.rnn_size, state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    # 初始化 state
    initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32);
    with tf.device("/GPU:0"):
        embedding = tf.get_variable('embedding', initializer=tf.random_uniform(
            [vocab_size + 1, cfg.rnn_size], -1.0, 1.0))
        inputs = tf.nn.embedding_lookup(embedding, X)

    # 使用循环网络
    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
    output = tf.reshape(outputs, [-1, cfg.rnn_size])

    weights = tf.Variable(tf.truncated_normal([cfg.rnn_size, vocab_size + 1]), dtype=tf.float32)
    bias = tf.Variable(tf.zeros(shape=[vocab_size + 1]), dtype=tf.float32)
    preds = (tf.matmul(output, weights) + bias)
    # 将标签转换成one_hot数据
    """
    0 0 0 1 0 0
    0 0 1 0 0 0
    """
    labels = tf.one_hot(tf.reshape(label, [-1]), depth=vocab_size + 1)
    # 计算loss值.
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=preds)
    # 求loss均值
    total_loss = tf.reduce_mean(loss)
    # 定义优化器,采用Adam
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    graph['initial_state'] = initial_state
    graph['output'] = output
    graph['train_op'] = train_op
    graph['total_loss'] = total_loss
    graph['loss'] = loss
    graph['last_state'] = last_state
    return graph


def predictPoems(X, vocab_size, num_layers=2, batch_size=1):
    """
    预测模块
    :param X: 输入的数据
    :param vocab_size: one_hot维度dim
    :param rnn_size:
    :param num_layers:
    :param batch_size: 默认为一首
    :return:
    """
    graph = {}
    if funcMap.get(cfg.model, None) is None:
        print ("type of the model is not existing ! you choose model type: {}".format(cfg.model))
        return None
    # cell表示选用的RNN内部核
    cell = funcMap[cfg.model](cfg.rnn_size, state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32);

    with tf.device("/GPU:0"):
        embedding = tf.get_variable('embedding', initializer=tf.random_uniform(
            [vocab_size + 1, cfg.rnn_size], -1.0, 1.0))
        inputs = tf.nn.embedding_lookup(embedding, X)

    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
    output = tf.reshape(outputs, [-1, cfg.rnn_size])
    weights = tf.Variable(tf.truncated_normal([cfg.rnn_size, vocab_size + 1]), dtype=tf.float32)
    bias = tf.Variable(tf.zeros(shape=[vocab_size + 1]), dtype=tf.float32)
    probs = (tf.matmul(output, weights) + bias)
    preds = tf.nn.softmax(probs)
    graph['initial_state'] = initial_state
    graph['last_state'] = last_state
    graph['prediction'] = preds

    return graph
