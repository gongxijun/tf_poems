# -*- coding:utf-8 -*-
"""
@author: gxjun
@file: trainPoems.py
@time: 17-9-26 上午11:52
"""

import os
import numpy as np
from random import randint
import tensorflow as tf
from model import model, predictPoems
from dataset import process_poems, generate_batch
from config import cfg
import argparse


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Poems Predict')
    parser.add_argument('--model_dir', dest='model_dir',
                        default='/media/gxjun/78289D37289CF4FA/py-faster-rcnn/tools/poems/output/poems',
                        help="model path")
    parser.add_argument('--data_dir', dest='data_dir',
                        default='/media/gxjun/78289D37289CF4FA/py-faster-rcnn/tools/poems/data/tangshi.txt',
                        help="data path ", type=str)
    parser.add_argument('--batch_size', dest='batch_size',
                        default=64,
                        help="batch of size every train step ", type=int)
    parser.add_argument('--max_epoch', dest='max_epoch', default=200, help='max epoch time', type=int)
    parser.add_argument('--learning_rate', dest='learning_rate', default=0.01, help='grad of learing_rate', type=int)

    return parser.parse_args()


args = parse_args();
poems_vector, ch_to_int, vocabularies = process_poems(args.data_dir)


def train():
    if cfg.debug:
        print('[debug] {}'.format(args.data_dir))
    if not os.path.exists(os.path.dirname(args.model_dir)):
        os.makedirs(os.path.dirname(args.model_dir))

    batches_inputs, batches_outputs = generate_batch(args.batch_size, poems_vector, ch_to_int)

    input_data = tf.placeholder(tf.int32, [args.batch_size, None])
    output_targets = tf.placeholder(tf.int32, [args.batch_size, None])

    end_points = model(model='lstm', input_data=input_data, output_data=output_targets, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=args.learning_rate)

    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer();
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    with tf.Session(config=config) as sess:
        sess.run(init)
        start_epoch = 1
        checkpoint = tf.train.latest_checkpoint(args.model_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("[INFO] restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
        print('[INFO] start training...')
        try:
            for epoch in range(start_epoch, args.max_epoch + 1):
                n_chunk = len(poems_vector) // args.batch_size
                for index, batch in enumerate(range(n_chunk)):
                    loss, _, _ = sess.run([
                        end_points['total_loss'],
                        end_points['last_state'],
                        end_points['train_op']
                    ], feed_dict={
                        input_data: batches_inputs[index],
                        output_targets: batches_outputs[index]
                    })

                    print('[INFO] Epoch: %d , batch: %d , training loss: %.6f' % (epoch, batch, loss))

                if epoch % 10 == 0:
                    saver.save(sess, os.path.join(args.model_dir, cfg.model_prefix), global_step=epoch);

        except KeyboardInterrupt:
            print('[INFO] Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, os.path.join(cfg.checkpoints_dir, cfg.model_prefix), global_step=epoch)
            print('[INFO] Last epoch were saved, next time will start from epoch {}.'.format(epoch))


def to_word(predict, vocabs):
    t = np.cumsum(predict)
    s = np.sum(predict)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    if sample > len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]


def cangtouPoems(words,input_data,end_points=None):
    print('[INFO] loading corpus from %s' % args.data_dir)

    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer();
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    with tf.Session(config=config) as sess:
        sess.run(init)

        checkpoint = tf.train.latest_checkpoint(args.model_dir)
        saver.restore(sess, checkpoint)

        x = np.array([list(map(ch_to_int.get, cfg.seq_begin_flag))])

        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                         feed_dict={input_data: x})
        # 解析关键字,
        words = words.split(',');
        if words is None or words.__len__() > 0:
            word = words[0];
        else:
            word = to_word(predict, vocabularies)
        poem = u''
        seq = 1;
        len_seq = 1
        while not (word == cfg.seq_end_flag and (len_seq > 0 and len_seq % 2 == 0)):
            if poem != u'' and u'。' == poem[-1]:
                if seq < len(words):
                    word = words[seq];
                    seq += 1;
                len_seq += 1;

            poem += word
            x = np.zeros((1, len(word)))
            if ch_to_int.get(word, None) is None:
                ch_to_int[word] = max(ch_to_int.values()) + 1;
            x[0, 0] = ch_to_int[word]
            [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                             feed_dict={input_data: x, end_points['initial_state']: last_state})
            word = to_word(predict, vocabularies)
        # word = words[np.argmax(probs_)]
        return poem


def tianchiPoems(words,input_data, end_points=None):
    print('[INFO] loading corpus from %s' % args.data_dir)

    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer();
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    with tf.Session(config=config) as sess:
        sess.run(init)

        checkpoint = tf.train.latest_checkpoint(args.model_dir)
        saver.restore(sess, checkpoint)

        x = np.array([list(map(ch_to_int.get, cfg.seq_begin_flag))])

        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                         feed_dict={input_data: x})
        # 解析关键字,
        words = words.split(',');
        poem = u''
        seq = 0;
        len_seq = 1;
        seq_pos = 0;
        word = to_word(predict, vocabularies)
        key_pos = -1;
        while not (word == cfg.seq_end_flag and (len_seq > 0 and len_seq % 2 == 0)):
            if key_pos == -1:
                key_pos = randint(1, 5);
            if poem != u'' and u'。' == poem[-1]:
                seq_pos = 0;
                len_seq += 1;
            else:
                seq_pos += 1;
            if key_pos == seq_pos and seq < len(words):
                word = words[seq];
                seq += 1;
                key_pos = -1;
            poem += word
            x = np.zeros((1, len(word)))
            if ch_to_int.get(word, None) is None:
                ch_to_int[word] = max(ch_to_int.values()) + 1;
            x[0, 0] = ch_to_int[word]
            [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                             feed_dict={input_data: x, end_points['initial_state']: last_state})
            word = to_word(predict, vocabularies)
        # word = words[np.argmax(probs_)]
        return poem


def pretty_print_poem(poem):
    poem_sentences = poem.split(u'。')
    if cfg.debug:
        print poem_sentences
    for s in poem_sentences:
        if s != u'' and len(s) > 10:
            print(s + u'。')


def run():
    print('[INFO] train tang poem...')
    train()

poem2=u'';
def predict():
    input_data = tf.placeholder(tf.int32, [1, None])

    end_points = predictPoems(X=input_data, vocab_size=len(
        vocabularies), num_layers=2, batch_size=1)
    while True:
        print('[INFO] 作诗一首')
        import sys
        print '_' * 35
        print '-' * 10 + '1 . 藏头诗' + '-' * 10
        print '-' * 10 + '2 . 填词写诗' + '-' * 10
        print '_' * 35
        choose_type = u'' + raw_input("选择类型:").decode(sys.stdin.encoding)
        key_words = u'' + raw_input("输入关键字,多个关键字以','分割").decode(sys.stdin.encoding)
        print key_words.split(',')
        if int(choose_type) ==2:
            poem2 = tianchiPoems(key_words,input_data,end_points)
        elif int(choose_type) ==1:
            poem2 = cangtouPoems(key_words,input_data,end_points)
        else:
            print "没有该选项：";
            continue;
        pretty_print_poem(poem2)


if __name__ == '__main__':
    run(); #如果训练
    predict() #如果预测
