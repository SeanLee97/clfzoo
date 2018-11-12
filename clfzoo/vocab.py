# -*- coding: utf-8 -*-

import numpy as np
from collections import OrderedDict

class Vocab(object):
    def __init__(self, vocab_path=None):
        # word
        self.idx2word = OrderedDict()
        self.word2idx = OrderedDict()
        self.word_cnt = OrderedDict()

        # char
        self.idx2char = OrderedDict()
        self.char2idx = OrderedDict()
        self.char_cnt = OrderedDict()

        self.label2idx = None  # assign value via dataloader
        self.idx2label = None  # 

        # embedding
        self.word_embed_dim = None
        self.word_embeddings = None
        
        self.char_embed_dim = None
        self.char_embeddings = None

        # init tokens
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'

        self.init_tokens = [self.pad_token, self.unk_token]

        for token in self.init_tokens:
            self.add_word(token)
            self.add_char(token)

        if vocab_path is not None:
            self.load_from_file(vocab_path)

    def load_file(self, fname):
        with open(fname, 'r') as rf:
            for line in rf:
                token = line.strip()
                if len(token) == 0:
                    continue
                self.add_word(token)
                [self.add_char(ch) for ch in token]

    def word_size(self):
        return len(self.idx2word)

    def char_size(self):
        return len(self.idx2char)
        
    def get_token_idx(self, token2idx, token):
        token = token.lower()
        return token2idx[token] \
                    if token in token2idx \
                    else token2idx[self.unk_token]

    def get_word_idx(self, token):
        return self.get_token_idx(self.word2idx, token)

    def get_char_idx(self, token):
        return self.get_token_idx(self.char2idx, token)

    def add_token(self, token2idx, idx2token, token_cnt, token, cnt=1):
        token = token.lower()
        if token in token2idx:
            idx = token2idx[token]
        else:
            idx = len(token2idx)
            token2idx[token] = idx
            idx2token[idx] = token

        if cnt > 0:
            if token in token_cnt:
                token_cnt[token] += 1
            else:
                token_cnt[token] = cnt

        return idx

    def add_word(self, token, cnt=1):
        self.add_token(self.word2idx, self.idx2word, self.word_cnt, token, cnt)

    def add_char(self, token, cnt=1):
        self.add_token(self.char2idx, self.idx2char, self.char_cnt, token, cnt)


    def filter_word_by_cnt(self, min_cnt):
        filtered_tokens = [token for token in self.word2idx if self.word_cnt[token] > min_cnt]
        self.word2idx = OrderedDict()
        self.idx2word = OrderedDict()
        self.char2idx = OrderedDict()
        self.idx2char = OrderedDict()

        for token in self.init_tokens:
            self.add_word(token, cnt=0)
            self.add_char(token, cnt=0)

        for token in filtered_tokens:
            self.add_word(token, cnt=0)
            [self.add_char(ch, cnt=0) for ch in token]

    def load_pretrained_word_embeddings(self, embedding_path, kernel='kv'):
        trained_embeddings = OrderedDict()
        if kernel == 'gensim':
            from gensim.models.word2vec import Word2Vec

            w2v_model = Word2Vec.load(embedding_path)
            word_dict = w2v_model.wv.vocab
            for token in word_dict:
                if token not in self.word2idx:
                    continue
                trained_embeddings[token] = w2v_model[token].tolist()
                if self.word_embed_dim is None:
                    self.word_embed_dim = len(list(trained_embeddings[token]))
        elif kernel == 'kv':
            import pickle

            with open(embedding_path, 'rb') as fin:
                word_dict = pickle.load(fin)
                for token in word_dict:
                    if token not in self.word2idx:
                        continue
                    trained_embeddings[token] = word_dict[token]
                    if self.word_embed_dim is None:
                        self.word_embed_dim = len(list(trained_embeddings[token]))
        else:
            raise NotImplementedError("Not support embedding kernel {}.".format(kernel))

        filtered_tokens = trained_embeddings.keys()

        self.word2idx = OrderedDict()
        self.id2token = OrderedDict()
        for token in self.init_tokens:
            self.add_word(token, cnt=0)
        for token in filtered_tokens:
            self.add_word(token, cnt=0)

        # load embeddings
        self.word_embeddings = np.random.rand(self.word_size(), self.word_embed_dim)
        for token in self.word2idx.keys():
            if token in trained_embeddings:
                self.word_embeddings[self.get_word_idx(token)] = trained_embeddings[token]

    def randomly_word_embeddings(self, embed_dim):
        self.word_embed_dim = embed_dim
        word_size= self.word_size()
        self.word_embeddings = np.random.rand(word_size, embed_dim)
        for token in self.init_tokens:
            self.word_embeddings[self.get_word_idx(token)] = np.zeros([embed_dim])

    def randomly_char_embeddings(self, embed_dim):
        self.char_embed_dim = embed_dim
        char_size = self.char_size()
        self.char_embeddings = np.random.rand(char_size, embed_dim)
        for token in self.init_tokens:
            self.char_embeddings[self.get_char_idx(token)] = np.zeros([embed_dim])

    def get_word_vector(self, tokens):
        vec = [self.get_word_idx(tok) for tok in tokens]
        return vec

    def get_char_vector(self, tokens):
        vec = []
        for token in tokens:
            char_vec = []
            for ch in token:
                char_vec.append(self.get_char_idx(ch))
            vec.append(char_vec)
        return vec


