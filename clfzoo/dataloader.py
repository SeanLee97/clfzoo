# -*- coding: utf-8 -*-

"""Dataloader
"""

import numpy as np
import os, re
import json

"""
Define your tokenizer
for Chinese jieba is the default tokenizer
"""
def tokenizer(sent):
    return sent.split()

def sentence_split(sent):
    return re.split('[.!?。！？]', sent)

class DataLoader(object):
    def __init__(self, config):
        """
        Args:
            config:
        """

        self.config = config
        self.logger = config.logger

        self.label2idx = {}
        self.idx2label = {}

        self.max_sent_num = config.max_sent_num
        self.max_sent_len = config.max_sent_len
        self.max_char_len = config.max_char_len

        self.train_set, self.dev_set, self.test_set = [], [], []

        if config.train_file is not None:
            self.logger.info("process train file {}".format(config.train_file))
            self.train_set = self._load_dataset(config.train_file, is_train=True)

        if config.dev_file is not None:
            self.logger.info("process dev file {}".format(config.dev_file))
            self.dev_set = self._load_dataset(config.dev_file, is_train=True)

        if config.test_file is not None:
            self.logger.info("process test file {}".format(config.test_file))
            self.test_set = self._load_dataset(config.test_file, is_train=False)

    def _load_dataset(self, fpath, is_train=False):
        with open(fpath, 'r') as rf:
            data_set = []
            label_set = set()
            for line in rf:
                line = line.strip()
                arr = line.split(self.config.splitter)
                if len(arr) == 0:
                    continue

                sample = {}                
                if is_train:
                    if len(arr) < 2:
                        continue
                    sample['label'] = arr[0].strip()
                    label_set.add(arr[0].strip())

                sample = self._get_sample(sample, arr[1])
                data_set.append(sample)
            
            if is_train:
                self.label2idx = dict(zip(label_set, range(len(label_set))))
                self.idx2label = dict(zip(range(len(label_set)), label_set))

        return data_set

    def set_data(self, vocab, datas, labels=None):

        data_set = []
        for idx, data in enumerate(datas):

            sample = {}                
            if labels is not None:
                sample['label'] = labels[idx]
            else:
                sample['label'] = 'unk'

            sample = self._get_sample(sample, data)

            sample['token_ids'] = []
            sample['token_char_ids'] = []
            for sentence in sample['tokens']:
                sample['token_ids'].append(vocab.get_word_vector(sentence))
                sample['token_char_ids'].append(vocab.get_char_vector(sentence))

            data_set.append(sample)

        self.predict_set = data_set

    def _get_sample(self, sample, sentence):
        if self.max_sent_num > 1:
            all_sentences = sentence_split(sentence)
            if len(all_sentences) > self.max_sent_num:
                sentences = all_sentences[:self.max_sent_num-1] + [''.join(all_sentences[self.max_sent_num-1:])]
            else:
                sentences = all_sentences

            sample['sentences'] = []
            sample['tokens'] = []
            sample['token_chars'] = []
            for sent in sentences:
                tokens = tokenizer(sent)
                sample['sentences'].append(sent)
                sample['tokens'].append(tokens)
                sample['token_chars'].append([list(token) for token in tokens])
        else:
            tokens = tokenizer(sentence)
            sample['sentences'] = [sentence]
            sample['tokens'] = [tokens]
            sample['token_chars'] = [[list(token) for token in tokens]]
        return sample

    def _mini_batch(self, batch_size, data, indices, pad_id):
        """
        Get mini batch
        Args:
            data: all data
            indices: the indices of samples to be selected
            pad_id:
        """

        batch_data = {'raw_data': [data[i] for i in indices],
                      'token_ids': [],
                      'token_char_ids': [],
                      'label_ids': []}

        for idx, sample in enumerate(batch_data['raw_data']):
            for sidx in range(self.max_sent_num):
                if sidx < len(sample['tokens']):
                    batch_data['token_ids'].append(sample['token_ids'][sidx])
                    batch_data['token_char_ids'].append(sample['token_char_ids'][sidx])
                else:
                    batch_data['token_ids'].append([])
                    batch_data['token_char_ids'].append([])

            batch_data['label_ids'].append(self.label2idx.get(sample['label'], 0))
        
        diff = batch_size - len(batch_data['label_ids'])
        if diff > 0:
            batch_data['label_ids'].extend([pad_id] * diff)

        batch_data = self._dynamic_padding(batch_size, batch_data, pad_id)
        return batch_data

    def _dynamic_padding(self, batch_size, batch_data, pad_id):
        """
        Dynamically pad the batch data with pad_id
        """
        pad_len = self.max_sent_len
        pad_char_len = self.max_char_len

        # pad token ids
        batch_data['token_ids'] = [(ids + [pad_id] * (pad_len - len(ids)))[:pad_len]
                                    for ids in batch_data['token_ids']]
        
        # pad char ids
        for index, char_list in enumerate(batch_data['token_char_ids']):
            for char_index in range(len(char_list)):
                if len(char_list[char_index]) >= pad_char_len:
                    char_list[char_index] = char_list[char_index][:pad_char_len]
                else:
                    char_list[char_index] += [pad_id]*(pad_char_len - len(char_list[char_index]))
            batch_data['token_char_ids'][index] = char_list

        batch_data['token_char_ids'] = [(ids + [[pad_id]*pad_char_len]*(pad_len-len(ids)))[:pad_len]
                                        for ids in batch_data['token_char_ids']]

        # pad full batch
        diff = batch_size - len(batch_data['token_ids'])
        if diff > 0:
            batch_data['token_ids'] += [[pad_id for y in range(pad_len)] for x in range(diff)]
            batch_data['token_char_ids']  += [[[pad_id for z in range(pad_char_len)] for y in range(pad_len)] for x in range(diff)]

        if self.max_sent_num > 1:
            token_ids = []
            token_char_ids = []
            sent_tokens = []
            sent_char_tokens = []
            for idx, token in enumerate(batch_data['token_ids'], 1):
                if idx % self.max_sent_num == 0:
                    token_ids.append(sent_tokens)
                    token_char_ids.append(sent_char_tokens)
                    sent_tokens = []
                    sent_char_tokens = []
                else:
                    sent_tokens.append(token)
                    sent_char_tokens.append(batch_data['token_char_ids'][idx])

            batch_data['token_ids'] = np.array(token_ids)
            batch_data['token_char_ids'] = np.array(token_char_ids)

            # padding batch
            shape = list(np.shape(batch_data['token_ids']))
            batch_diff = batch_size - shape[0]
            if batch_diff > 0:
                a_token_ids = np.zeros([batch_diff, shape[1], pad_len])
                batch_data['token_ids'] = np.append(batch_data['token_ids'], a_token_ids, axis=0)

                a_char_token_ids = np.zeros([batch_diff, shape[1], pad_len, pad_char_len])
                batch_data['token_char_ids'] = np.append(batch_data['token_char_ids'], a_char_token_ids, axis=0)

            # padding sent num
            diff = self.max_sent_num - list(np.shape(batch_data['token_ids']))[1]
            if diff > 0:
                a_token_ids = np.zeros([batch_size, diff, pad_len])
                batch_data['token_ids'] = np.append(batch_data['token_ids'], a_token_ids, axis=1)

                a_char_token_ids = np.zeros([batch_size, diff, pad_len, pad_char_len])
                batch_data['token_char_ids'] = np.append(batch_data['token_char_ids'], a_char_token_ids, axis=1)

        return batch_data

    def next_batch(self, setname, batch_size, pad_id, shuffle=False):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            setname: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if setname == 'train':
            data = self.train_set
        elif setname == 'dev':
            data = self.dev_set
        elif setname == 'test':
            data = self.test_set
        elif setname == 'predict':
            data = self.predict_set
        else:
            raise NotImplementedError("No dataset named as {}".format(setname))

        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        
        for start_id in np.arange(0, data_size, batch_size):
            batch_indices = indices[start_id: start_id + batch_size]
            yield self._mini_batch(batch_size, data, batch_indices, pad_id)

    def convert_to_ids(self, vocab):
        """
        convert the data from text to ids
        Args:
            vocab: 
        """

        for dataset in [self.train_set, self.dev_set, self.test_set]:
            if dataset is None:
                continue
            for sample in dataset:
                sample['token_ids'] = []
                sample['token_char_ids'] = []
                for sentence in sample['tokens']:
                    sample['token_ids'].append(vocab.get_word_vector(sentence))
                    sample['token_char_ids'].append(vocab.get_char_vector(sentence))

    def word_iter(self, setname=None):
        """
        Iterates over all the words in the dataset
        Args:
            setname
        Return:
            a generator
        """

        if setname is None:
            dataset = self.train_set + self.dev_set + self.test_set
        elif setname == 'train':
            dataset = self.train_set
        elif setname == 'dev':
            dataset = self.dev_set
        elif setname == 'test':
            dataset = self.test_set
        elif setname == 'predict':
            dataset = self.predict_set
        else:
            raise NotImplementedError("No dataset named as{}".format(setname))

        if dataset is not None:
            for sample in dataset:
                for sent in sample['tokens']:
                    for token in sent:
                         yield token
