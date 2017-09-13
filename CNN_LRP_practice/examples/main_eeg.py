# -*- coding: utf-8 -*-
'''
@author: Vignesh Srinivasan
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@maintainer: Vignesh Srinivasan
@maintainer: Sebastian Lapuschkin
@contact: vignesh.srinivasan@hhi.fraunhofer.de
@date: 20.12.2016
@version: 1.0+
@copyright: Copyright (c)  2016-2017, Vignesh Srinivasan, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.append("..")
from modules.sequential import Sequential
from modules.linear import Linear
from modules.softmax import Softmax
from modules.relu import Relu
from modules.tanh import Tanh
from modules.convolution import Convolution
from modules.avgpool import AvgPool
from modules.maxpool import MaxPool
from modules.utils import Utils, Summaries, plot_relevances
import modules.render as render
import inputeeg2

import tensorflow as tf
import numpy as np
import pdb
import scipy.io as sio

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("max_steps", 100, 'Number of steps to run trainer.')  # 5001
flags.DEFINE_integer("batch_size", 100, 'Number of steps to run trainer.')  # 1000
flags.DEFINE_integer("test_every", 1, 'Number of steps to run trainer.')  # 500
flags.DEFINE_float("learning_rate", 0.001, 'Initial learning rate')  # 0.01
flags.DEFINE_float("dropout", 0.8, 'Keep probability for training dropout.')  # 0.9
flags.DEFINE_string("data_dir", 'data', 'Directory for storing data')
flags.DEFINE_string("summaries_dir", 'eeg_convolutional_logs', 'Summaries directory')
flags.DEFINE_boolean("relevance", True, 'Compute relevances')  # 임의로 변경함
flags.DEFINE_string("relevance_method", 'simple', 'relevance methods: simple/eps/w^2/alphabeta')
flags.DEFINE_boolean("save_model", False, 'Save the trained model')
flags.DEFINE_boolean("reload_model", False, 'Restore the trained model')
# flags.DEFINE_string("checkpoint_dir", 'eeg_convolution_model','Checkpoint dir')
flags.DEFINE_string("checkpoint_dir", 'eeg_trained_model', 'Checkpoint dir')
flags.DEFINE_string("checkpoint_reload_dir", 'eeg_trained_model', 'Checkpoint dir')

# skyeom's own setting
flags.DEFINE_integer("input_x_dim", 9, 'x-size of spatial map')
flags.DEFINE_integer("input_y_dim", 11, 'y-size of spatial map')
flags.DEFINE_integer("input_z_dim", 7, 'size of time-series')

FLAGS = flags.FLAGS


def nn():
    return Sequential([Convolution(output_depth=12, input_depth=3, batch_size=FLAGS.batch_size,
                                   input_x_dim=FLAGS.input_x_dim, input_y_dim=FLAGS.input_y_dim,
                                   input_z_dim=FLAGS.input_z_dim,
                                   kernel_size=5, act='relu', stride_size=2, pad='SAME'),
                       MaxPool(),

                       Convolution(kernel_size=2, output_depth=16, stride_size=2, act='relu', pad='SAME'),
                       MaxPool(),

                       Convolution(kernel_size=2, output_depth=24, stride_size=1, act='relu', pad='SAME'),
                       MaxPool(),

                       Convolution(kernel_size=2, output_depth=32, stride_size=1, pad='SAME'),
                       Convolution(kernel_size=1, output_depth=2, stride_size=1, pad='SAME')])


trX, trY, teX, teY = inputeeg2.train_data, inputeeg2.train_labels, inputeeg2.test_data, inputeeg2.test_labels


def train():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        # Input placeholders
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, 9, 11, 7, 3], name='x-input')
            y_ = tf.placeholder(tf.float32, [None, 2], name='y-input')
            keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope('model'):
            net = nn()
            inp = tf.reshape(x, [FLAGS.batch_size, FLAGS.input_x_dim, FLAGS.input_y_dim, FLAGS.input_z_dim, 3])
            op = net.forward(inp)
            y = tf.squeeze(op)
        # print(inp)
        # Back-propagated NN learning 하는 곳
        # trainer = net.fit(output=y, ground_truth=y_, loss='softmax_crossentropy', optimizer='adam',
        #				  opt_params=[FLAGS.learning_rate])

        with tf.variable_scope('relevance'):
            if FLAGS.relevance:
                LRP = net.lrp(op, FLAGS.relevance_method, 1e-8)

                # LRP layerwise
                relevance_layerwise = []
            # R = y
            # for layer in net.modules[::-1]:
            #	 R = net.lrp_layerwise(layer, R, 'simple')
            #	 relevance_layerwise.append(R)

            else:
                LRP = []
                relevance_layerwise = []

        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))

        tf.summary.scalar('accuracy', accuracy)

        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

        tf.global_variables_initializer().run()

        utils = Utils(sess, FLAGS.checkpoint_reload_dir)
        if FLAGS.reload_model:
            utils.reload_model()

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cost)
        predict_op = tf.argmax(y, 1)

        tf.initialize_all_variables().run()
        for i in range(FLAGS.max_steps):
            for bnum in xrange(int(len(trX) / FLAGS.batch_size)):
                t_trX = trX[bnum * FLAGS.batch_size:(bnum + 1) * FLAGS.batch_size, :, :, :, :]
                t_trY = trY[bnum * FLAGS.batch_size:(bnum + 1) * FLAGS.batch_size, :]
                sess.run(train_op, feed_dict={x: t_trX, y_: t_trY, keep_prob: 1})
                t_teX = teX[bnum * FLAGS.batch_size:(bnum + 1) * FLAGS.batch_size, :, :, :, :]
                t_teY = teY[bnum * FLAGS.batch_size:(bnum + 1) * FLAGS.batch_size, :]
                print(i * int(len(trX) / FLAGS.batch_size) + bnum, np.mean(
                    np.argmax(t_teY, axis=1) == sess.run(predict_op, feed_dict={x: t_teX, y_: t_teY, keep_prob: 1})))

        for bnum in xrange(int(len(trX) / FLAGS.batch_size)):
            t_teX = teX[bnum * FLAGS.batch_size:(bnum + 1) * FLAGS.batch_size, :, :, :, :]
            t_teY = teY[bnum * FLAGS.batch_size:(bnum + 1) * FLAGS.batch_size, :]
            test_inp = {x: t_teX, y_: t_teY, keep_prob: 1}
            relevance_test = sess.run(LRP, feed_dict={x: t_teX, y_: t_teY, keep_prob: 1})

        if FLAGS.relevance:
            sio.savemat('/relevance_test_sub1.mat', {"relevance_test": relevance_test})
            #relevance_test = relevance_test[:, 0:9, 0:11, 0:3, :]
            # plot test images with relevances overlaid
            images = test_inp[test_inp.keys()[0]].reshape([FLAGS.batch_size, 9, 11, 7, 3])
            # images = (images + 1)/2.0
            plot_relevances(relevance_test.reshape([FLAGS.batch_size, 9, 11, 7, 3]), images, test_writer)

def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()

if __name__ == '__main__':
    tf.app.run()
