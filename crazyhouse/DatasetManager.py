import numpy as np
import chess
import chess.pgn
from python.CrazyhouseGame import CrazyhouseGame as Game
from crazyhouse.NNet import NNetWrapper as nn
import threading
import concurrent.futures
import pickle
from random import shuffle

class DatasetManager:

    def __init__(self):
        self.file_paths = ["../data/training/lichess_db_crazyhouse_rated_2021-07.pgn",
                               "../data/training/lichess_db_crazyhouse_rated_2021-08.pgn",
                               "../data/training/lichess_db_crazyhouse_rated_2021-09.pgn",
                               "../data/training/lichess_db_crazyhouse_rated_2021-10.pgn",
                               "../data/training/lichess_db_crazyhouse_rated_2021-11.pgn",
                               "../data/training/lichess_db_crazyhouse_rated_2021-12.pgn",
                               "../data/training/lichess_db_crazyhouse_rated_2022-01.pgn",
                               "../data/training/lichess_db_crazyhouse_rated_2022-02.pgn"]
        self.num_games = 0
        self.lock = threading.Lock()

    def filter_data(self, file_path):
        pgn_file = open(file_path)
        game_info = chess.pgn.read_game(pgn_file)

        games = list()

        num_abandoned = 0
        num_low_elo = 0

        while game_info:
            with self.lock:
                self.num_games += 1

            if game_info.headers["Termination"] not in ["Normal", "Time forfeit"]:
                num_abandoned += 1
            elif int(game_info.headers["WhiteElo"]) < 2000 or int(game_info.headers["BlackElo"]) < 2000:
                num_low_elo += 1
            else:
                games.append(game_info)

            game_info = chess.pgn.read_game(pgn_file)

        return games

    def filter_dataset(self, file_path):
        with concurrent.futures.ThreadPoolExecutor() as e:
            futures = list()
            for path in self.file_paths:
                futures.append(e.submit(self.filter_data, path))

            done = False
            while not done:
                print(self.num_games, end='\r')

                done_count = 0
                for future in futures:
                    if future.done():
                        done_count += 1

                if done_count == 8:
                    done = True

            print()
            print("Exporting data to " + file_path + "...")

            self.num_games = 0

            file = open(file_path, "w")
            for future in futures:
                for game_info in future.result():
                    self.num_games += 1
                    print(game_info, file=file, end="\n\n")
                    print(self.num_games, end="\r")

            print()
            print("Done!")
