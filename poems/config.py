# -*- coding:utf-8 -*-
"""
@author: gxjun
@file: config.py
@time: 17-9-27 下午5:18
"""
# coding:utf-8
__author__ = 'xijun.gong'
import os
import os.path as osp
from easydict import EasyDict as edict

__C = edict()
cfg = __C
# 诗词开始和结束的标志
__C.seq_begin_flag = 'G'
__C.seq_end_flag = 'E'
# default 默认为lstm模型
__C.model = 'lstm'
__C.rnn_size =128
__C.debug =True
__C.model_prefix='poems'