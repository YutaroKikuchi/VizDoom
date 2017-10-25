#!/usr/bin/env python

from __future__ import print_function
from vizdoom import *

from random import choice
from time import sleep

import tensorflow as tf

resolution = (120,120)

class VP_Network(object):
    def __init__(self):

        self.testdata = [[10 for i in range(120)] for j in range(120)]

        self.s1_ = tf.placeholder(tf.float32,[None] + list(resolution)+[1],name="state")
        #self.a_ = tf.placeholder(tf.int32, [None],name="partition")

        self.conv1 = tf.contrib.layers.convolution2d(self.s1_, num_outputs=60, kernel_size=[3,3],stride=[1,1],
        activation_fn=tf.nn.relu,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        biases_initializer=tf.constant_initializer(0.1))
        
        self.max_pool1 = tf.contrib.layers.max_pool2d(self.conv1,kernel_size=[2,2])
        
        self.conv2 = tf.contrib.layers.convolution2d(self.max_pool1, num_outputs=60, kernel_size=[3,3],stride=[1,1],
        activation_fn=tf.nn.relu,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        biases_initializer=tf.constant_initializer(0.1))
        
        self.max_pool2 = tf.contrib.layers.max_pool2d(self.conv2,kernel_size=[2,2])

        self.fc1 = tf.contrib.layers.fully_connected(self.max_pool2,num_outputs=40,activation_fn=tf.nn.relu,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        biases_initializer=tf.constant_initializer(0.1))

        self.fc2 = tf.contrib.layers.fully_connected(self.fc1,num_outputs=5,activation_fn=tf.nn.relu,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        biases_initializer=tf.constant_initializer(0.1))

        self.output = tf.contrib.layers.softmax(self.fc2)

    def test(self):
        sess = tf.Session()

        print(sess.run(self.output,feed_dict={self.s1_: [self.testdata]}))


network = VP_Network()

network.test()
