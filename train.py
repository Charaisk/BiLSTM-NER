#!/usr/bin/env python 
# -*- coding:utf-8 -*-
'''
@author:HE
@date:2019/9/9
@name:train.py
@IDE:PyCharm
'''
import tensorflow as tf
from data_utils import TextConverter, read_corpus, batch_generator
from model import BilstmNer
import os

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('name', 'default', 'name of the model')
tf.flags.DEFINE_integer('batch_size', 64, 'number of seqs in one batch')
tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
tf.flags.DEFINE_float('train_keep_prob', 0.5, 'dropout rate during training')
tf.flags.DEFINE_string('input_file', './data/train_data', 'utf8 encoded text file')
tf.flags.DEFINE_integer('max_steps', 100000, 'max steps to train')
tf.flags.DEFINE_integer('save_every_n', 1000, 'save the model every n steps')
tf.flags.DEFINE_integer('log_every_n', 10, 'log to the screen every n steps')
tf.flags.DEFINE_integer('max_vocab', 3500, 'max char number')


def main(_):
    model_path = os.path.join('model', FLAGS.name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    data = read_corpus(FLAGS.input_file)
    converter = TextConverter(data, FLAGS.max_vocab)
    converter.save_to_file(os.path.join(model_path, 'converter.pkl'))

    g = batch_generator(data, FLAGS.batch_size)
    print(converter.vocab_size)
    model = BilstmNer(converter.vocab_size,
                      converter.num_classes,
                      lstm_size=FLAGS.lstm_size,
                      learning_rate=FLAGS.learning_rate,
                      train_keep_prob=FLAGS.train_keep_prob,
                      embedding_size=FLAGS.embedding_size
                      )
    model.train(g,
                FLAGS.max_steps,
                model_path,
                FLAGS.save_every_n,
                FLAGS.log_every_n,
                )


if __name__ == '__main__':
    tf.app.run()