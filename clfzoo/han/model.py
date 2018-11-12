# -*- coding: utf-8 -*-

import tensorflow as tf

import clfzoo.libs as libs
import clfzoo.libs.loss as loss
from clfzoo.libs.rnn import *
from clfzoo.libs.attention.multihead_attention import MultiheadAttention
from clfzoo.base import BaseModel

class HAN(BaseModel):
    def __init__(self, vocab, config):
        super(HAN, self).__init__(config)

        self.vocab = vocab
        self.n_classes = len(vocab.label2idx)

        self.build_graph()

    def build_graph(self):
        self.layer_placeholder()
        self.layer_embedding()
        self.layer_encoder()
        self.layer_predict()

        self.init_session()

    def layer_placeholder(self):
        """
        Define placeholders
        """
        self.s = tf.placeholder(tf.int32, 
                                [None, self.config.max_sent_num, self.config.max_sent_len],
                                name="sentence")
        self.label = tf.placeholder(tf.int32,
                                    [None], name="label")

        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")

    def layer_embedding(self):
        """
        Layer embedding
        Default fuse word and char embeddings
        """
        with tf.variable_scope('embeddings'):
            self.word_mat = tf.get_variable(
                                'word_embeddings',
                                shape=[self.vocab.word_size(), self.vocab.word_embed_dim],
                                initializer=tf.constant_initializer(self.vocab.word_embeddings),
                                trainable=False)

            self.s_emb = tf.nn.embedding_lookup(self.word_mat, self.s)

    def self_attention(self, values, scope="self_attention", reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            Q = values
            K = values
            V = values

            shape = V.get_shape().as_list()

            W = tf.get_variable('attn_W', 
                                shape=[shape[-1], shape[-1]],
                                #initializer=tf.contrib.layers.xavier_initializer(),
                                initializer=tf.truncated_normal_initializer(stddev=0.1),
                                dtype=tf.float32)
            U = tf.get_variable('attn_U',
                                shape=[shape[-1], shape[-1]],
                                #initializer=tf.contrib.layers.xavier_initializer(),
                                initializer=tf.truncated_normal_initializer(stddev=0.1),
                                dtype=tf.float32)
            P = tf.get_variable('attn_P',
                                shape=[shape[-1], 1],
                                #initializer=tf.contrib.layers.xavier_initializer(),
                                initializer=tf.truncated_normal_initializer(stddev=0.1),
                                dtype=tf.float32)

            Q = tf.reshape(Q, [-1, shape[-1]])
            K = tf.reshape(K, [-1, shape[-1]])

            similarity = tf.nn.tanh(
                      tf.multiply(
                          tf.matmul(Q, W),
                          tf.matmul(K, U)))

            alpha = tf.nn.softmax(tf.reshape(tf.matmul(similarity, P), [-1, shape[1], 1]), axis=1)
            V_t = tf.multiply(alpha, V)
        return V_t


    def layer_encoder(self):
        """
        Layer Encoder
        """
        with tf.variable_scope('encoder'):
            shape = self.s_emb.get_shape().as_list()

            if self.config.gpu >= 0:
                rnn_kernel = CudnnRNN
            else:
                rnn_kernel = RNN

            with tf.variable_scope("word_level_encoder"):
                rnn = rnn_kernel(self.config.hidden_dim, self.config.batch_size, shape[-1],
                                 dropout=self.dropout, kernel='gru')

                #attention = MultiheadAttention(num_heads=self.config.num_heads, dropout=self.dropout)

                sent_embeds = [tf.squeeze(x) for x in tf.split(self.s_emb, self.config.max_sent_num, axis=1)]

                outputs = []
                for i in range(self.config.max_sent_num):
                    sent = sent_embeds[i]
                    sent = tf.reshape(sent, [-1, self.config.max_sent_len, shape[-1]])
                    
                    is_reuse= True if i>0 else False
                    word_encoder, _ = rnn(sent, scope="word_level", reuse=is_reuse)

                    word_attention = self.self_attention(word_encoder, scope="word_attention", reuse=is_reuse) 
                    word_attention = tf.reduce_sum(word_attention, axis=1)
                    #word_attention = tf.contrib.layers.layer_norm(word_attention)
                    
                    outputs.append(word_attention)

            with tf.variable_scope("sentence_level_encoder"):
                sentence = tf.stack(outputs, axis=1) #[batch_size,num_sentence,num_units*2]
                shape = sentence.get_shape()

                rnn = rnn_kernel(self.config.hidden_dim, shape[0], shape[-1],
                                 dropout=self.dropout, kernel='gru')

                sentence_encoder, _ = rnn(sentence, scope="sentence_level")
                sentence_attention = self.self_attention(sentence_encoder, scope="sentence_attention")
                sentence_attention = tf.contrib.layers.layer_norm(sentence_attention)
                self.output = tf.reduce_sum(sentence_attention, axis=1) 
                #self.output = sentence_attention[:, -1, :]

 
    def layer_predict(self):
        with tf.variable_scope('predict'):

            output = tf.layers.batch_normalization(self.output)
            h = tf.layers.dense(output, self.config.hidden_dim)
            h = tf.nn.relu(h)

            self.logit = tf.layers.dense(h, self.n_classes)            
            
            if self.config.loss_type == 'cross_entropy':
                self.loss = loss.cross_entropy(self.label, self.logit)
            elif self.config.loss_type == 'focal_loss':
                self.loss = loss.focal_loss(self.label, self.logit)
            else:
                raise NotImplementedError("No loss type named {}".format(self.config.loss_type))

            self.predict = tf.argmax(tf.nn.softmax(self.logit), axis=1)
            self.probs = tf.nn.top_k(tf.nn.softmax(self.logit), 1)[0]

            true_pred = tf.equal(tf.cast(self.label, tf.int64), self.predict)
            self.accu = tf.reduce_mean(tf.cast(true_pred, tf.float32))
            # train op
            self.add_train_op(self.loss)

    def run_epoch(self, train, dev, epoch):
        """
        train epoch

        Args:
            train: train dataset
            dev: dev dataset
            epoch: current epoch
        """ 

        total_loss, total_accu = 0.0, 0.0

        for idx, batch in enumerate(train, 1):
            feed_dict = {
                self.s: batch['token_ids'],
                self.label: batch['label_ids'],
                self.dropout: self.config.dropout,
            }

            try:
                _, loss, accu = self.sess.run([self.train_op, self.loss, self.accu], feed_dict)
                total_loss += loss
                total_accu += accu 
            except Exception as e:
                continue

            if idx % self.config.log_per_batch == 0:
                self.logger.info("Average loss from batch {} to {} is {}, accuracy is {}".format(
                    idx-self.config.log_per_batch+1, idx, 
                    total_loss/self.config.log_per_batch,
                    total_accu/self.config.log_per_batch 
                ))
                total_loss = 0
                total_accu = 0

        # evaluate dev
        _, metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)

        return metrics

    def run_evaluate(self, dev):
        y_true, y_pred, y_score = [], [], []

        total_loss = 0.0
        total_num = 0

        for idx, batch in enumerate(dev, 1):
            feed_dict = {
                self.s: batch['token_ids'],
                self.label: batch['label_ids'],
                self.dropout: 0.0,
            }

            try:
                predict, prob, loss = self.sess.run([self.predict, self.probs, self.loss], feed_dict)
                total_loss += loss
                total_num += 1

                real_size = len(batch['raw_data'])
                y_pred += predict.tolist()[:real_size]
                y_score += prob.tolist()[:real_size]
                y_true += batch['label_ids'][:real_size]
            except Exception as e:
                print("Error>>>", e)
                continue

        y_pred_labels = [self.vocab.idx2label[lidx] for lidx in y_pred]
        metrics = self.calc_metrics(y_pred, y_true)
        return list(zip(y_pred_labels, y_score)), metrics


