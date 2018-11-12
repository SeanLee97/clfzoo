# -*- coding: utf-8 -*-

import sys
sys.path.append('../..')

import clfzoo
import clfzoo.transformer as clf
from clfzoo.config import ConfigTransformer

class Config(ConfigTransformer):
    def __init__(self):
        super(Config, self).__init__()

    batch_size = 8
    
    epochs = 50
    max_sent_num = 1
    max_sent_len = 50
    
    lr_rate=0.0001
    num_units=128
    num_heads=6
    num_blocks=4
 
    train_file = '../data/english/TREC.train.txt'
    dev_file = '../data/english/TREC.test.txt'

clf.model(Config(), training=True)
clf.train()
