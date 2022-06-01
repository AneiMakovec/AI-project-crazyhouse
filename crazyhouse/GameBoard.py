import copy
from chess import *
from chess.variant import *
import typing

class GameBoard(chess.variant.CrazyhouseBoard):

    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1"

    def __init__(self, fen: Optional[str] = starting_fen, chess960: bool = False) -> None:
        super().__init__(fen, chess960=chess960)
        self.state_repetitions = {}
        state, _ = super().fen().split(None, 1)
        self.state_repetitions[state] = 0

    def push(self, move: chess.Move) -> None:
        # make move
        super().push(move)

        state, _ = self.fen().split(None, 1)
        if state not in self.state_repetitions:
            self.state_repetitions[state] = 0
        else:
            self.state_repetitions[state] += 1

    def copy(self: CrazyhouseBoardT, stack: Union[bool, int] = True) -> CrazyhouseBoardT:
        board = super().copy(stack=stack)
        board.state_repetitions = copy.deepcopy(self.state_repetitions)
        # board.move_count = self.move_count
        return board

    def mirror(self: CrazyhouseBoardT) -> CrazyhouseBoardT:
        board = super().mirror()
        board.state_repetitions = copy.deepcopy(self.state_repetitions)
        # board.move_count = self.move_count
        self.mirrored = True
        return board
