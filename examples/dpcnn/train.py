# -*- coding: utf-8 -*-

import sys
sys.path.append('../..')

import clfzoo
import clfzoo.dpcnn as clf
from clfzoo.config import ConfigDPCNN

class Config(ConfigDPCNN):
    def __init__(self):
        super(Config, self).__init__()

    epochs = 20
    batch_size = 8

    max_sent_num = 1
    max_sent_len = 60
    max_char_len = 10

    train_file = '../data/english/TREC.train.txt'
    dev_file = '../data/english/TREC.test.txt'


clf.model(Config(), training=True)
clf.train()
