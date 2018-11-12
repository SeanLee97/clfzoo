# -*- coding: utf-8 -*-

import os
import pickle
from clfzoo.dataloader import DataLoader
from clfzoo.vocab import Vocab

class Instance(object):
    def __init__(self, config, training=False):
        self.logger = config.logger

        self.dataloader = DataLoader(config)
        if training:
            self.logger.info("Preprocesing...")
            self.vocab = Vocab()
            self.vocab.label2idx = self.dataloader.label2idx
            self.vocab.idx2label = self.dataloader.idx2label

            for word in self.dataloader.word_iter('train'):
                self.vocab.add_word(word)
                [self.vocab.add_char(ch) for ch in word]

            unfiltered_vocab_size = self.vocab.word_size()
            self.vocab.filter_word_by_cnt(config.min_word_freq)
            filtered_num = unfiltered_vocab_size - self.vocab.word_size()
            self.logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num, self.vocab.word_size()))
            
            self.logger.info('Assigning embeddings...')
            if config.use_pretrained_embedding  and config.pretrained_embedding_file is not None:
                self.logger.info("Load pretrained word embedding...")
                self.vocab.load_pretrained_word_embeddings(config.pretrained_embedding_file, kernel=config.embedding_kernel)
            else:
                self.vocab.randomly_word_embeddings(config.word_embed_dim)
        
            self.vocab.randomly_char_embeddings(config.char_embed_dim)

            self.logger.info('Saving vocab...')
            with open(os.path.join(config.vocab_dir, 'vocab.data'), 'wb') as fout:
                pickle.dump(self.vocab, fout)

            self.logger.info('====== Done with preparing! ======')
        else:
            self.logger.info("Load prev vocab......")
            with open(os.path.join(config.vocab_dir, 'vocab.data'), 'rb') as fin:
                self.vocab = pickle.load(fin)
                self.dataloader.label2idx = self.vocab.label2idx
                self.dataloader.idx2label = self.vocab.idx2label

    def set_data(self, datas, labels=None):
        self.dataloader.set_data(self.vocab, datas, labels)
