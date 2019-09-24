#!/usr/bin/env python 
# -*- coding:utf-8 -*-
'''
@author:HE
@date:2019/9/24
@name:main.py
@IDE:PyCharm
'''
import tensorflow as tf
from data_utils import TextConverter
from model import BilstmNer
import os

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_string('converter_path', 'model/default/converter.pkl', 'model/name/converter.pkl')
tf.flags.DEFINE_string('checkpoint_path', 'model/default', 'checkpoint path')
tf.flags.DEFINE_integer('max_length', 30, 'max length to generate')


def main(_):
    converter = TextConverter(filename=FLAGS.converter_path)
    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path = \
            tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    model = BilstmNer(converter.vocab_size, converter.num_classes,
                      lstm_size=FLAGS.lstm_size,
                      embedding_size=FLAGS.embedding_size)
    print("[*] Success to read {}".format(FLAGS.checkpoint_path))
    model.load(FLAGS.checkpoint_path)

    demo_sent = "京剧研究院就美日联合事件讨论"
    tag = model.demo([(converter.text_to_arr(demo_sent), [0]*len(demo_sent))])
    print(tag)


if __name__ == '__main__':
    tf.app.run()
