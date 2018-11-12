# -*- coding: utf-8 -*-

import sys
sys.path.append('../..')

import clfzoo
import clfzoo.han as clf
from clfzoo.config import ConfigHAN

class Config(ConfigHAN):
    def __init__(self):
        super(Config, self).__init__()
    
    gpu = -1

    batch_size = 16
    epochs = 50
    lr_rate = 1e-3

    max_sent_num = 25
    max_sent_len = 60

    train_file = '../data/english/TREC.train.txt'
    dev_file = '../data/english/TREC.test.txt'

clf.model(Config(), training=True)
clf.train()
