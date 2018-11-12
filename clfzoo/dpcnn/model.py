# -*- coding: utf-8 -*-

import tensorflow as tf

import clfzoo.libs as libs
import clfzoo.libs.loss as loss
from clfzoo.libs.conv import Conv
from clfzoo.libs.highway import Highway
from clfzoo.base import BaseModel

class DPCNN(BaseModel):
    def __init__(self, vocab, config):
        super(DPCNN, self).__init__(config)

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
                                [None, self.config.max_sent_len],
                                name="sentence")

        self.sh = tf.placeholder(tf.int32,
                                 [None, self.config.max_sent_len, self.config.max_char_len],
                                 name="sentence_char")

        self.label = tf.placeholder(tf.int32,
                                    [None], name="label")

        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        
        self.s_mask = tf.cast(self.s, tf.bool)
        self.s_len = tf.reduce_sum(tf.cast(self.s_mask, tf.int32), axis=1)
        self.sh_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.sh, tf.bool), tf.int32), axis=2), [-1])

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
            self.char_mat = tf.get_variable(
                                'char_embeddins',
                                shape=[self.vocab.char_size(), self.vocab.char_embed_dim],
                                initializer=tf.constant_initializer(self.vocab.char_embeddings),
                                trainable=False)

            sh_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.sh), 
                                                       [self.config.batch_size*self.config.max_sent_len, 
                                                        self.config.max_char_len, 
                                                        self.vocab.char_embed_dim])
            # projection
            sh_emb = Conv(bias=True)(sh_emb, self.config.filter_size, scope="sh_conv")
            sh_emb = tf.reduce_max(sh_emb, axis=1)
            sh_emb = tf.reshape(sh_emb, [self.config.batch_size, self.config.max_sent_len, -1])

            s_emb = tf.nn.embedding_lookup(self.word_mat, self.s)
            s_emb = tf.concat([s_emb, sh_emb], axis=2)

            # projection
            s_emb = Conv(bias=True)(s_emb, self.config.filter_size, scope="s_proj")
            self.s_emb = Highway(activation=tf.nn.relu, 
                                 kernel='conv', 
                                 dropout=self.dropout)(s_emb, scope='highway')

    def layer_encoder(self):
        """
        Layer Encoder
        Multi-channels convolution to encode
        """
        with tf.variable_scope('encoder'):
            inputs = self.s_emb

            conv1 = Conv(bias=True, activation=tf.nn.relu, kernel_size=3)(inputs, self.config.filter_size, scope='conv1')
            conv2 = Conv(bias=True, activation=tf.nn.relu, kernel_size=3)(conv1, self.config.filter_size, scope='conv2')

            # shortcut
            conv2 = conv2 + inputs

            inputs = conv2

            for i in range(self.config.num_blocks):
                seq_len = inputs.get_shape()[1]
                poolings = tf.transpose(
                    tf.nn.top_k(tf.transpose(inputs, [0,2,1]), k=seq_len // 2)[0], [0, 2, 1])

                conv1 = Conv(bias=True, activation=tf.nn.relu, kernel_size=3)(poolings, self.config.filter_size, scope='conv1-%d' % i)
                conv2 = Conv(bias=True, activation=tf.nn.relu, kernel_size=3)(conv1, self.config.filter_size, scope='conv2-%d' % i) 

                inputs = conv2 + poolings

            self.output = tf.reduce_max(inputs, axis=1)
 
    def layer_predict(self):
        with tf.variable_scope('predict'):

            output = tf.layers.batch_normalization(self.output)
            h = tf.layers.dense(output, self.config.filter_size//2)
            h = tf.tanh(h)
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
            #print(batch['token_char_ids'])
            feed_dict = {
                self.s: batch['token_ids'],
                self.sh: batch['token_char_ids'],
                self.label: batch['label_ids'],
                self.dropout: self.config.dropout,
            }

            try:
                _, loss, accu = self.sess.run([self.train_op, self.loss, self.accu], feed_dict)
                total_loss += loss
                total_accu += accu 
            except Exception as e:
                print(e)
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
                self.sh: batch['token_char_ids'],
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

