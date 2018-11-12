# -*- coding: utf-8 -*-

"""Base class for general models.
"""

import os
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# clear session
tf.keras.backend.clear_session()

class BaseModel(object):

    def __init__(self, config):
        self.config = config
        self.logger = config.logger
        self.sess = None
        self.saver = None

        self.lr = self.config.lr_rate

    def train(self, dataloader, pad_id=0):
        """
        Args:
            train: training dataset that yield tuple (word_idx, label_idx)
            dev: develope dataset that yield tuple (word_idx, label_idx)
        """
        best_score = 0
        no_improve_epoch = 0

        self.add_summary()

        for epoch in range(self.config.epochs):
            train = dataloader.next_batch('train', self.config.batch_size, pad_id, shuffle=True)
            dev = dataloader.next_batch('dev', self.config.batch_size, pad_id, shuffle=False)

            self.logger.info("Epoch {} / {}".format(epoch+1, self.config.epochs)) 
            metrics = self.run_epoch(train, dev, epoch)

            #print(">>>>", best_score, metrics[self.config.eval_metric])

            if best_score <= metrics[self.config.eval_metric]:
                best_score = metrics[self.config.eval_metric]
                no_improve_epoch = 0
                self.save()
                self.logger.info("new best score!")
            else:
                no_improve_epoch += 1

            if self.config.lr_decay > 0:
                self.lr *= self.config.lr_decay

            if self.config.early_stop > 0 and no_improve_epoch > self.config.early_stop:
                self.logger.info("early stopping {} epochs without improvement".format(no_improve_epoch))
                break  
    
    def evaluate(self, test):
        """
        Evaluate model on test set
        """
        return self.run_evaluate(test)

    def calc_metrics(self, pred_labels, true_labels):
        return {
            'accu': accuracy_score(true_labels, pred_labels),
            'p': precision_score(true_labels, pred_labels, average='macro'),
            'r': recall_score(true_labels, pred_labels, average='macro'),
            'f1': f1_score(true_labels, pred_labels, average='macro')
        }
        

    def add_train_op(self, loss):
        optim_type = self.config.optimizer.lower()
        with tf.variable_scope("train_scope"):
            if optim_type == 'adam':
                optimizer = tf.train.AdamOptimizer(self.lr)
            elif optim_type == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.lr)
            elif optim_type == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(self.lr)
            elif optim_type == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.lr)
            elif optim_type == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.lr)
            else:
                raise NotImplementedError("Optimizer {} is not support".format(optim_type))
            
            if self.config.clipper > 0:
                grads, vs     = zip(*optimizer.compute_gradients(loss))
                grads, gnorm  = tf.clip_by_global_norm(grads, self.config.clipper)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)

    def init_session(self):
        """
        Initialize session
        """
        self.logger.info("Init session")
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

    def close_session(self):
        """
        close session
        """
        self.sess.close()

    def add_summary(self):
        self.merged = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.config.graph_dir, self.sess.graph)
    
    def save(self):
        """
        Saves the model into model_dir with model_name as the model indicator
        """
        if not os.path.exists(self.config.model_dir):
            os.makedirs(self.config.model_dir)
        self.saver.save(self.sess, os.path.join(self.config.model_dir, self.config.model_name))
        self.logger.info('Model saved in {}, with name {}.'.format(self.config.model_dir, self.config.model_name))

    def restore(self):
        """
        Restores the model into model_dir from model_name as the model indicator
        """
        self.saver.restore(self.sess, os.path.join(self.config.model_dir, self.config.model_name))
        self.logger.info('Model restored from {}, with prefix {}'.format(self.config.model_dir, self.config.model_name))

