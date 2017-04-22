import tensorflow as tf
from tfs import *



class AModelWithDropout(BaseModel):

    def __init__(self, dataset):

        self.is_training = tf.placeholder(tf.bool, [])
        self.drop_pr = tf.placeholder(tf.float32, [])

        defaults = {
            'conv'  : {
                'window'        : 3,
                'stride'        : 1,
                'act'           : 'relu',
                'batch_norm'    : True,
                'training'      : self.is_training
            },
            'pool'  : {
                'window'        : 2,
                'stride'        : 2
            },
            'dense' : {
                'units'         : 2000,
                'act'           : 'relu'
            }
        }

        net = [
            ['conv', {
                'chan'  : 32
            }],
            ['pool'],
            ['conv', {
                'chan'  : 64
            }],
            ['conv', {
                'chan'  : 64
            }],
            ['pool'],
            ['conv', {
                'chan'  : 128
            }],
            ['conv', {
                'chan'  : 128
            }],
            ['pool'],
            ['conv', {
                'chan'  : 256
            }],
            ['conv', {
                'chan'  : 256
            }],
            ['pool'],
            ['conv', {
                'chan'  : 256
            }],
            ['conv', {
                'chan'  : 256
            }],
            ['pool'],
            ['flatten'],
            ['dense'],
            ['dropout', {
                'pr'    : self.drop_pr
            }],
            ['dense', {
                'units' : 1000
            }]
        ]

        self.logits_train = sequential(dataset.train['images'], net, defaults = defaults, name = 'model_with_dropout')
        self.logits_val = sequential(dataset.val['images'], net, defaults = defaults, name = 'model_with_dropout', reuse = True)
        args = {
            'y'             : dataset.train['labels'],
            'y_pred'        : self.logits_train,
            'y_val'         : dataset.val['labels'],
            'y_pred_val'    : self.logits_val,
            'rate'          : FLAGS.rate
        }
        self.optimizer, train_summary, val_summary, self.global_step = classify(**args)
        self.train_summary_op   = tf.summary.merge(train_summary)
        self.val_summary_op    = tf.summary.merge(val_summary)

        model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for i in model_vars:
            tf.summary.histogram(i.op.name, i)
        self.summary_op =  tf.summary.merge_all()

    def train(self, sess):
        return sess.run([self.optimizer, self.train_summary_op, self.global_step], feed_dict = {self.is_training: True, self.drop_pr: 0.6})[1:]

    def validate(self, sess):
        return sess.run(self.summary_op, feed_dict = {self.is_training: False, self.drop_pr: 1.0})




class VGG16(BaseModel):

    def __init__(self, dataset):

        self.is_training = tf.placeholder(tf.bool, [])

        defaults = {
            'conv'  : {
                'window'        : 3,
                'stride'        : 1,
                'act'           : 'relu',
                'batch_norm'    : True,
                'training'      : self.is_training
            },
            'pool'  : {
                'window'        : 2,
                'stride'        : 2
            },
            'dense' : {
                'units'         : 4096,
                'act'           : 'relu'
            }
        }

        net = [
            ['conv', {
                'chan'  : 64
            }],
            ['conv', {
                'chan'  : 64
            }],
            ['pool'],
            ['conv', {
                'chan'  : 128
            }],
            ['conv', {
                'chan'  : 128
            }],
            ['pool'],
            ['conv', {
                'chan'  : 256
            }],
            ['conv', {
                'chan'  : 256
            }],
            ['conv', {
                'chan'  : 256
            }],
            ['pool'],
            ['conv', {
                'chan'  : 512
            }],
            ['conv', {
                'chan'  : 512
            }],
            ['conv', {
                'chan'  : 512
            }],
            ['pool'],
            ['conv', {
                'chan'  : 512
            }],
            ['conv', {
                'chan'  : 512
            }],
            ['conv', {
                'chan'  : 512
            }],
            ['pool'],
            ['flatten'],
            ['dense'],
            ['dense'],
            ['dense', {
                'units' : 1000
            }]
        ]

        self.logits_train = sequential(dataset.train['images'], net, defaults = defaults, name = 'vgg16')
        self.logits_val = sequential(dataset.val['images'], net, defaults = defaults, name = 'vgg16', reuse = True)
        args = {
            'y'             : dataset.train['labels'],
            'y_pred'        : self.logits_train,
            'y_val'        : dataset.val['labels'],
            'y_pred_val'   : self.logits_val,
            'rate'          : FLAGS.rate
        }
        self.optimizer, train_summary, val_summary, self.global_step = classify(**args)
        self.train_summary_op   = tf.summary.merge(train_summary)
        self.val_summary_op    = tf.summary.merge(val_summary)

        model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for i in model_vars:
            tf.summary.histogram(i.op.name, i)
        self.summary_op =  tf.summary.merge_all()



    def train(self, sess):
        return sess.run([self.optimizer, self.train_summary_op, self.global_step], feed_dict = {self.is_training: True})[1:]

    def validate(self, sess):
        return sess.run(self.summary_op, feed_dict = {self.is_training: False})


