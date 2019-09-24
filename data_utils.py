#!/usr/bin/env python 
# -*- coding:utf-8 -*-
'''
@author:HE
@date:2019/9/18
@name:data_utils.py
@IDE:PyCharm
'''
import pickle
import numpy as np
import copy

save_path = "./model/default/converter.pkl"
corpus_path = "./data/train_data"


def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent, tag = [], []
    for line in lines:
        if line != '\n':
            char, label = line.strip().split()
            sent.append(char)
            tag.append(label)
        else:
            data.append((sent, tag))
            sent, tag = [], []

    return data


def pad_sequences(sequences, pad_mark=0):
    """

    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x: len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_generator(data, batch_size):  # 根据输入的arr，返回对应的生成器，满足输入的序列个数和序列长度
    data = np.array(copy.copy(data))
    n_batches = len(data) // batch_size  # 一共可以得到多少组输入数据
    data = data[:batch_size * n_batches]  # 直接忽略了后面不能构成一组输入的数据
    data = data.reshape((-1, batch_size, 2))
    converter = TextConverter(filename=save_path)
    while True:
        np.random.shuffle(data)  # 将所有行打乱顺序
        for n in range(0, n_batches):
            lines = data[n:n + 1, :]  # 每次选择对应n_seqs行，n_steps列的数据
            x = [converter.text_to_arr(lines[0][i][0]) for i in range(batch_size)]
            y = [converter.label_to_id(lines[0][i][1]) for i in range(batch_size)]
            yield x, y


class TextConverter(object):
    def __init__(self, data=None, max_vocab=5000, filename=None):
        """

        :param corpus_path:
        :param max_vocab:
        :param filename:
        """
        if filename is not None:
            with open(filename, 'rb') as f:
                self.vocab = pickle.load(f)
        else:
            vocab_count = {}  # 存储每一个数据单元在整个读入的文本中出现的次数的字典
            for sent, _ in data:
                for word in sent:
                    if word.isdigit():
                        word = '<NUM>'
                    elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                        word = '<ENG>'
                    vocab_count[word] = vocab_count.get(word, 0) + 1
            vocab_count_list = []  # 存储元组(数据单元，对应数量)组成的列表，然后按照数量的大小排序，比如[('a',100),('d',20),...,('x',3)]
            for word in vocab_count:
                vocab_count_list.append((word, vocab_count[word]))
            vocab_count_list.sort(key=lambda x: x[1], reverse=True)
            if len(vocab_count_list) > max_vocab:  # 根据传入的最大数量的数据单元数截断前max_vocab大的数据单元，基本上不可能，除非遇到汉字之类的文本
                vocab_count_list = vocab_count_list[:max_vocab]
            vocab = [x[0] for x in vocab_count_list]
            self.vocab = vocab  # vocab仅仅存储数据单元按照出现数量从大到小的列表，例如：['a','d',...,'x']

        self.tag2label = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "B-ORG": 5, "I-ORG": 6}
        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}  # 数据单元到数字字典{' ':0,'e':1,...,'c':20,...}
        self.int_to_word_table = dict(enumerate(self.vocab))  # 数字到数据单元字典{0:‘ ’，1:'e',...,20:'c',..}

    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    @property
    def num_classes(self):
        return len(self.tag2label)

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)  # 如果出现了没有出现的词，则变为<unk>对应的标记

    def int_to_word(self, index):
        if index == len(self.vocab):
            return '<Unk>'  # 没有出现的词被标记为unknown的缩写
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')

    def text_to_arr(self,
                    text):  # 将输入的text根据word_to_int返回得到对应的编码数，并构成np.ndarray并返回，例如：输入' a\n'，则返回类似array([ 0, 0, 4, 10])
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return np.array(arr)

    def arr_to_text(self, arr):  # 输入列表类型的数据，返回对应的数据单元的组合
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)

    def label_to_id(self, label):  # 输入列表类型的标签，返回对应的id的组合
        id = []
        for tag in label:
            id.append(self.tag2label[tag])
        return np.array(id)

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f)
