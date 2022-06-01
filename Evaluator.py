import chess
import chess.pgn
import chess.engine

# from crazyhouse.python.CrazyhouseGame import CrazyhouseGame as Game
from crazyhouse.GameBoard import GameBoard

STOCKFISH_PATH = r"D:\Development\SDK\stockfish_15_win_x64_avx2\stockfish_15_x64_avx2.exe"
FAIRY_STOCKFISH_PATH = r"D:\Development\SDK\fairy_stockfish\fairy-stockfish-largeboard_x86-64-bmi2.exe"

class Evaluator():

    def __init__(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(FAIRY_STOCKFISH_PATH)

    def evaluate_games(self, file_path):
        board = GameBoard()

        pgn_file = open(file_path)
        game_info = chess.pgn.read_game(pgn_file)

        while game_info:
            for move in game_info.mainline_moves():
                move_evals = {}
                best_eval = -float('inf')
                for legal_move in board.legal_moves:
                    info = self.engine.analyse(board, chess.engine.Limit(time=1), root_moves=[legal_move])
                    eval = int(str(info["score"].relative))
                    if eval > best_eval:
                        best_eval = eval
                    move_evals[legal_move.uci()] = eval

                print(f"move eval: {move_evals[move.uci()]}, best eval: {best_eval}")
                board.push(move)

            input()
            game_info = chess.pgn.read_game(pgn_file)
        self.engine.quit()


# board = GameBoard("rnbqkbnr/pppp2pp/8/5p2/8/3P4/PPP1PP1P/RNBQKBNR[Pp] w KQkq - 0 5")
# # #Now make sure you give the correct location for your stockfish engine file
# # #...in the line that follows by correctly defining path
# engine = chess.engine.SimpleEngine.popen_uci(FAIRY_STOCKFISH_PATH)
# # for key in engine.options:
# #     if key == "UCI_Variant":
# #         print(engine.options[key])
#
# # if board.turn: print('White to move')
# # else: print('black to move')
# #
# for el in board.legal_moves:
#     info = engine.analyse(board, chess.engine.Limit(time=1), root_moves=[el])
#     t = int(str(info["score"].relative))
# #     # if t.startswith('#'):
# #     #         print(str(board.san(el))," eval = mate in ", t)
# #     # else: print(str(board.san(el))," eval = ", round(int(t)/100.,2))
#     print(el.uci())
#     print(t)
#
# # for key in engine.options:
# #     if key == "UCI_Variant":
# #         print(engine.options[key])
# engine.quit()

# print(pyffish.legal_moves("crazyhouse", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1", []))

# eval = Evaluator()
# eval.evaluate_games(r"D:\Development\Projects\CrazyZero python\data\training\training_set.pgn")
