import os
import logging

import coloredlogs

from Coach import Coach
from Trainer import Trainer
from crazyhouse.python.CrazyhouseGame import CrazyhouseGame as Game
# from crazyhouse.cpp.game import GameCpp as Game
from crazyhouse.GameBoard import GameBoard
from crazyhouse.NNet import NNetWrapper as nn
from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 10,
    'numEps': 50,                # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 100,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 10,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'cpuct_init': 2.5,
    'cpuct_base': 19652,
    'u_min': 0.25,
    'u_init': 1.0,
    'u_base': 1965,
    'Q_thresh_init': 0.5,
    'Q_thresh_max': 0.9,
    'Q_thresh_base': 1965,
    'Q_factor': 0.7,
    'check_thresh': 0.1,
    'check_factor': 0.5,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./checkpoint/','human_data_100%.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    'num_threads': 5

})


def main():
    print("Operating modes:")
    print("     1. Train with human data")
    print("     2. Train with self-play")
    print()
    mode = input("Select mode: ")

    try:
        mode = int(mode)
    except:
        print("Please insert only the number of the selected mode.")
        exit(0)

    if mode == 1:
        num_games = input("Insert number of games to train with: ")

        try:
            num_games = int(num_games)
        except:
            print("Please insert only the number of games.")
            exit(0)

        log.info("Starting %s...", Trainer.__name__)

        t = Trainer()

        t.train_with_games("./data/training/training_set.pgn", num_games)
    else:
        log.info('Loading %s...', Game.__name__)
        g = Game()

        log.info('Loading %s...', nn.__name__)
        nnet = nn(g)

        if args.load_model:
            log.info('Loading checkpoint...')
            nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
        else:
            log.warning('Not loading a checkpoint!')

        log.info('Loading the Coach...')
        c = Coach(g, nnet, args)

        #if args.load_model:
        #    log.info("Loading 'trainExamples' from file...")
        #    c.loadTrainExamples()

        log.info('Starting the learning process 🎉')
        c.learn()


if __name__ == "__main__":
    main()
