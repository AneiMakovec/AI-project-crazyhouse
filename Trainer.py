import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

import chess
import chess.pgn

from crazyhouse.python.CrazyhouseGame import CrazyhouseGame as Game
from crazyhouse.NNet import NNetWrapper as nn
from crazyhouse.GameBoard import GameBoard

log = logging.getLogger(__name__)

LOAD_GAMES = 100

class Trainer():

    def train_with_games(self, games_file_path, num_train_games):
        # read all the games from the file
        pgn_file = open(games_file_path)
        game_info = chess.pgn.read_game(pgn_file)

        game = Game()
        nnet = nn(game)

        # return None

        all_games = 0
        num_games = 0
        train_set = list()

        display_text = True

        while game_info:
            if num_games < LOAD_GAMES and all_games < num_train_games:
                # replay each game and store the input representations of every game state
                if display_text:
                    log.info("Loading train data...")
                    display_text = False

                board = GameBoard()

                res = 0.0
                if game_info.headers["Result"] == "1-0":
                    res = 1.0
                elif game_info.headers["Result"] == "0-1":
                    res = -1.0

                # print(game_info.headers["Result"])

                player = True

                for move in game_info.mainline_moves():
                    # board = game.getCanonicalForm(board, 1)

                    # if not player:
                    #     move = chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square), move.promotion, move.drop)
                    #     board.fullmove_number += 1
                    #     v = -res
                    # else:
                    #     v = res
                    if board.turn:
                        v = res
                    else:
                        v = -res

                    # print(board)
                    # print(v)
                    # input()

                    action = game.encodeAction(move)
                    pi = np.zeros(game.getActionSize())
                    pi[action] = 1.0
                    input_rep = game.inputRepresentation(board)

                    train_set.append((input_rep, pi, v))

                    board, _ = game.getNextState(board, 1, action)

                    # player = not player

                num_games += 1
                all_games += 1
                game_info = chess.pgn.read_game(pgn_file)
            else:
                # train the network
                # print()
                log.info("Started training...")
                # print()

                num_games = 0
                display_text = True

                shuffle(train_set)
                nnet.train(train_set)

                train_set.clear()

                log.info("Trained on " + str(all_games) + " in total.")
                # print()

                if all_games >= num_train_games:
                    break

        # print()

        # save the current neural network
        filename = "human_data_" + str(num_train_games) + ".pth.tar"
        nnet.save_checkpoint(filename=filename)
        log.info("Done!")
        # print()

    # def train_with_games(self, games_file_path, num_train_games):
    #     # read all the games from the file
    #     pgn_file = open(games_file_path)
    #     game_info = chess.pgn.read_game(pgn_file)
    #
    #     num_games = 0
    #
    #     games = list()
    #     while game_info:
    #         games.append(game_info)
    #         num_games += 1
    #         print("Loading games: " + str(num_games), end="\r")
    #         game_info = chess.pgn.read_game(pgn_file)
    #
    #     print()
    #
    #     shuffle(games)
    #     games = games[0:num_train_games]
    #
    #     game = Game()
    #     nnet = nn(game)
    #
    #     num_games = 0
    #     all_games = 0
    #     train_set = list()
    #
    #     for game_info in games:
    #         if num_games < 100:
    #             # replay each game and store the input representations of every game state
    #             board = GameBoard()
    #
    #             res = 0.0
    #             if game_info.headers["Result"] == "1-0":
    #                 res = 1.0
    #             elif game_info.headers["Result"] == "0-1":
    #                 res = -1.0
    #
    #             for move in game_info.mainline_moves():
    #                 action = game.encodeAction(move)
    #                 v = res if board.turn else -res
    #                 pi = np.zeros(game.getActionSize())
    #                 pi[action] = 1.0
    #                 input_rep = game.inputRepresentation(board)
    #
    #                 train_set.append((input_rep, pi, v))
    #                 board, _ = game.getNextState(board, 1, action)
    #
    #             num_games += 1
    #             all_games += 1
    #             print("Loading train data: " + str(num_games), end="\r")
    #         else:
    #             # train the network
    #             print()
    #             print("Started training...")
    #
    #             num_games = 0
    #
    #             shuffle(train_set)
    #             nnet.train(train_set)
    #
    #             train_set = list()
    #
    #             print("Trained on " + str(all_games) + " in total.")
    #
    #     print()
    #
    #     # save the current neural network
    #     filename = "human_data_" + str(num_train_games) + ".pth.tar"
    #     nnet.save_checkpoint(filename=filename)
    #     print("Done!")
