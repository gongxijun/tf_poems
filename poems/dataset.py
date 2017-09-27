#-*- coding:utf-8 -*-
"""
@author: gxjun
@file: paresPoems.py
@time: 17-9-26 上午11:52
"""

import collections
import os
import sys
import numpy as np
import codecs
import config

def process_poems(file_name):
    # 诗集
    poems = []
    with codecs.open(file_name, "r",encoding='utf-8') as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(u':')
                # print '   {}   '.format(title);
                # print content
                content = content.replace(u' ', u'')
                if u'_' in content or u'(' in content or u'（' in content or u'《' in content or u'[' in content or \
                        config.cfg.seq_begin_flag in content or config.cfg.seq_end_flag in content:
                    continue
                #print len(content)
                # if u'□□' in content:
                #     print u'   {}   '.format(title);
                #     print content
                #     continue;
                if len(content) < 5 or len(content) > 79:
                    continue
                content = config.cfg.seq_begin_flag + content + config.cfg.seq_end_flag
                poems.append(content)
            except ValueError as e:
                print e
                pass
    # 按诗的字数排序
    poems = sorted(poems, key=lambda l: len(line))

    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    # 这里根据包含了每个字对应的频率
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)

    # 取前多少个常用字
    words = words[:len(words)] + (u' ',)
    # 每个字映射为一个数字ID
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(lambda word: word_int_map.get(word, len(words)), poem)) for poem in poems]

    return poems_vector, word_int_map, words


def generate_batch(batch_size, poems_vec, word_to_int):
    # 每次取64首诗进行训练
    n_chunk = len(poems_vec) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size

        batches = poems_vec[start_index:end_index]
        # 找到这个batch的所有poem中最长的poem的长度
        length = max(map(len, batches))
        # 填充一个这么大小的空batch，空的地方放空格对应的index标号
        x_data = np.full((batch_size, length), word_to_int[' '], np.int32)
        for row in range(batch_size):
            # 每一行就是一首诗，在原本的长度上把诗还原上去
            x_data[row, :len(batches[row])] = batches[row]
        y_data = np.copy(x_data)
        # y的话就是x向左边也就是前面移动一个
        y_data[:, :-1] = x_data[:, 1:]
        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches