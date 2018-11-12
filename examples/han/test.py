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
    lr_rate = 1e-4
    max_sent_num = 25
    max_sent_len = 60
    train_file = '../data/english/TREC.train.txt'
    dev_file = '../data/english/TREC.test.txt'


clf.model(Config())

datas = ['几点了', '天气怎么样啊？']
labels = ['datetime', 'weather']
preds, metrics = clf.test(datas, labels)
print(preds)
print(metrics)
