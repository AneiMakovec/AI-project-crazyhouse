import sys
sys.path.append('..')
from utils import *

import os
import argparse
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.activations import *
from tensorflow.keras.regularizers import *

def value_loss_fn(y_true, y_pred):
    return 0.01 * tf.square(y_true - y_pred)

def policy_loss_fn(y_true, y_pred):
    return -tf.math.reduce_sum(tf.math.multiply(y_true, tf.math.log(y_pred)), 1)

class CrazyhouseNNet():
    def __init__(self, args):
        # game params
        # self.board_x, self.board_y = game.getBoardSize()
        # self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        # Inputs
        self.input_boards = Input(shape=(34,64))
        inputs = Reshape((34, 8, 8))(self.input_boards)

        conv0 = Conv2D(args.num_channels, kernel_size=3, strides=1, padding="same", data_format="channels_first", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(inputs)
        bn0 = BatchNormalization()(conv0)
        t = Activation('relu')(bn0)

        for i in range(self.args.num_residual_layers):
            convX = Conv2D(128 + 64 * i, kernel_size=1, strides=1, padding="same", data_format="channels_first", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(t)
            bnX = BatchNormalization()(convX)
            reluX = Activation('relu')(bnX)

            convX = DepthwiseConv2D(kernel_size=3, strides=1, padding="same", data_format="channels_first", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(reluX)
            bnX = BatchNormalization()(convX)
            reluX = Activation('relu')(bnX)

            convX = Conv2D(256, kernel_size=1, strides=1, padding="same", data_format="channels_first", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(reluX)
            bnX = BatchNormalization()(convX)

            t = Add()([bnX, t])

        # value head
        value_head = Conv2D(8, kernel_size=1, strides=1, padding="same", data_format="channels_first", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(t)
        value_head = BatchNormalization()(value_head)
        value_head = Activation('relu')(value_head)

        value_head = Flatten()(value_head)
        value_head = Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(value_head)
        value_head = Activation('relu')(value_head)

        value_head = Dense(1, activation='tanh', name='v')(value_head)

        # policy head
        policy_head = Conv2D(256, kernel_size=3, strides=1, padding='same', data_format="channels_first", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(t)
        policy_head = BatchNormalization()(policy_head)
        policy_head = Activation('relu')(policy_head)

        policy_head = Conv2D(81, kernel_size=3, strides=1, padding='same', data_format="channels_first", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(policy_head)
        policy_head = Flatten()(policy_head)
        policy_head = Dense(5184, activation='softmax', name='pi')(policy_head)


        self.pi = policy_head
        self.v = value_head

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=[policy_loss_fn, value_loss_fn], optimizer=SGD(learning_rate=0.00001, momentum=0.95, nesterov=True))
