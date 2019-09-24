#!/usr/bin/env python 
# -*- coding:utf-8 -*-
'''
@author:HE
@date:2019/9/5
@name:model.py
@IDE:PyCharm
'''
# coding: utf-8
from __future__ import print_function
from tensorflow.contrib.crf import crf_log_likelihood, viterbi_decode
from data_utils import pad_sequences, TextConverter
import tensorflow as tf
import time
import os


class BilstmNer:
    def __init__(self, vocab_size, num_classes, lstm_size=128,
                 learning_rate=0.001, grad_clip=5,
                 train_keep_prob=0.5, embedding_size=128):

        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.lstm_size = lstm_size
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.embedding_size = embedding_size

        tf.reset_default_graph()
        self.build_inputs()
        self.build_lstm()
        self.build_loss()
        self.build_optimizer()
        self.saver = tf.train.Saver()

    def build_inputs(self):
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32, shape=(None, None)
                                         , name='inputs')
            self.targets = tf.placeholder(tf.int32, shape=(None, None)
                                          , name='targets')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.sequence_lengths = tf.placeholder(tf.int32, name="sequence_lengths")

            with tf.device("/cpu:0"):
                embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size])
                self.lstm_inputs = tf.nn.embedding_lookup(embedding, self.inputs)

    def build_lstm(self):
        # 创建双向cells
        with tf.name_scope('bi_lstm'):
            cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size)
            cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size)
            (cell_fw_outputs, cell_bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.lstm_inputs,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)

            # 通过lstm_outputs得到概率
            seq_output = tf.concat([cell_fw_outputs, cell_bw_outputs], axis=-1)
            seq_output = tf.nn.dropout(seq_output, keep_prob=self.keep_prob)
            seq_shape = tf.shape(seq_output)
            self.pred_input = tf.reshape(seq_output, [-1, 2 * self.lstm_size])

            with tf.variable_scope('proj'):
                proj_w = tf.Variable(tf.truncated_normal([2 * self.lstm_size, self.num_classes], stddev=0.1))
                proj_b = tf.Variable(tf.zeros(self.num_classes))

            self.pred = tf.matmul(self.pred_input, proj_w) + proj_b
            self.logits = tf.reshape(self.pred, [-1, seq_shape[1], self.num_classes])

    def build_loss(self):
        with tf.name_scope('crf_loss'):
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits, tag_indices=self.targets,
                                                                        sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood)

    def build_optimizer(self):
        # 使用clipping gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))

    def train(self, batch_generator, max_steps, save_path, save_every_n, log_every_n):
        self.session = tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            # Train network
            step = 0
            for x, y in batch_generator:
                inputs, sequence_lengths = pad_sequences(x, pad_mark=0)
                targets, _ = pad_sequences(y, pad_mark=0)
                step += 1
                start = time.time()
                feed = {self.inputs: inputs,
                        self.targets: targets,
                        self.sequence_lengths: sequence_lengths,
                        self.keep_prob: self.train_keep_prob}
                batch_loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed)

                end = time.time()
                # control the print lines
                if step % log_every_n == 0:
                    print('step: {}/{}... '.format(step, max_steps),
                          'loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))
                if (step % save_every_n == 0):
                    self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
                if step >= max_steps:
                    break
            self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)

    def predict(self, inputs, sequence_lengths):
        sess = self.session
        feed = {self.inputs: [inputs],
                self.sequence_lengths: sequence_lengths,
                self.keep_prob: 1}
        logits, transition_params = sess.run([self.logits, self.transition_params], feed_dict=feed)
        labels = []
        for logit, sequence_length in zip(logits, sequence_lengths):
            viterbi_seq, _ = viterbi_decode(score=logit[:sequence_length], transition_params=transition_params)
            labels.append(viterbi_seq)
        return labels

    def demo(self, data):
        """

        :param sess:
        :param sent:
        :return:
        """
        label_list = []
        for seqs, _ in data:
            _, sequence_lengths = pad_sequences([seqs])
            label_list_ = self.predict(seqs, sequence_lengths)
            label_list.extend(label_list_)
        label2tag = {}
        tag2label = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "B-ORG": 5, "I-ORG": 6}
        for tag, label in tag2label.items():
            label2tag[label] = tag if label != 0 else label
        tag = [label2tag[label] for label in label_list[0]]
        return tag

    def load(self, checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))
