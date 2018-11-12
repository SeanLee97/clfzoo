# -*- coding: utf-8 -*-

import sys
sys.path.append('../..')

import clfzoo
import clfzoo.textrnn as clf
from clfzoo.config import ConfigTextRNN

class Config(ConfigTextRNN):
    def __init__(self):
        super(Config, self).__init__()

    gpu = -1

    epochs = 20
    batch_size = 8
    max_sent_num = 1
    max_sent_len = 60
    train_file = '../data/english/TREC.train.txt'
    dev_file = '../data/english/TREC.test.txt'

clf.model(Config(), training=True)
clf.train()
