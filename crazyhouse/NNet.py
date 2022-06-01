import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
sys.path.append('..')
from utils import *
# import plotly.graph_objects as graphs
# from crazyhouse.clr import OneCycleLR

import logging
import coloredlogs
log = logging.getLogger(__name__)

import argparse

from CrazyhouseNNet import CrazyhouseNNet as cnnet
# from .AlphaZeroNNet import AlphaZeroNNet as aznnet

CRAZYHOUSE_NNET = 13
ALPHAZERO_NNET = 19

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 7,
    'batch_size': 150,
    'cuda': True,
    'num_channels': 256,
    'num_residual_layers': CRAZYHOUSE_NNET
})

class NNetWrapper():
    def __init__(self, game):
        if args.num_residual_layers == CRAZYHOUSE_NNET:
            self.nnet = cnnet(args)
        else:
            self.nnet = aznnet(args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.game = game

    def train(self, examples):
        """
        This function trains the neural network with examples obtained from
        self-play.

        Input:
            examples: a list of training examples, where each example is of form
                      (board, pi, v). pi is the MCTS informed policy vector for
                      the given board, and v is its value. The examples has
                      board in its canonical form.
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)

        # lr_manager = OneCycleLR(len(input_boards), args.batch_size, 0.35,
        #                 end_percentage=0.66, scale_percentage=0.1,
        #                 maximum_momentum=0.95, minimum_momentum=0.8)

        # hist = self.nnet.model.fit(x = input_boards, y = [target_pis, target_vs], batch_size = args.batch_size, epochs = args.epochs, callbacks=[lr_manager])
        hist = self.nnet.model.fit(x = input_boards, y = [target_pis, target_vs], batch_size = args.batch_size, epochs = args.epochs)
        # self.history.append(hist.history)
        # self.display_training(hist.history)

    def predict(self, board):
        """
        Input:
            board: current board in its canonical form.

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float in [-1,1] that gives the value of the current board
        """
        # timing
        #start = time.time()
        #board = self.game.vectorize_board(board)
        # preparing input
        input_rep = self.game.inputRepresentation(board)
        input_rep = input_rep[np.newaxis, :, :]

        # run
        pi, v = self.nnet.model.predict(input_rep)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        # if not os.path.exists(filepath):
        #     print("No model in path {}".format(filepath))
        #     return
        self.nnet.model.load_weights(filepath)
        log.info('Loading Weights...')

    def display_training(self, history):
        plot = graphs.Figure()
        plot.add_trace(graphs.Scattergl(y=history['loss'], name='Train loss'))
        plot.add_trace(graphs.Scattergl(y=history['pi_loss'], name='Pi loss'))
        plot.add_trace(graphs.Scattergl(y=history['v_loss'], name='Value loss'))
        plot.update_layout(height=500, width=700, xaxis_title='Epoch', yaxis_title='Loss')
        plot.show()
