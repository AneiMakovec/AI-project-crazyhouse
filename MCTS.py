import logging
import math
import time

import numpy as np

EPS = 1e-8
INITIAL_Q = -1.0

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s
        # self.Ws = {}  # stores boards s where an action that leads to a winning terminal node was found
        self.player = 0

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        # if there is only one move available, return it
        valid_actions = np.argwhere(self.game.getValidMoves(canonicalBoard, 1) > 0.0).flatten()
        if len(valid_actions) == 1:
            probs = np.zeros(self.game.getActionSize())
            probs[valid_actions[0]] = 1.0
            return probs

        s = self.game.stringRepresentation(canonicalBoard)

        start = time.time()
        for i in range(self.args.numMCTSSims):
            winning_node = self.search(canonicalBoard.copy())
            if winning_node:
                break

        # get number of visits for each move
        counts = np.array([self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())])

        # calculate the Q threshold
        Q_thresh = self.args.Q_thresh_max - math.exp(-self.Ns[s] / self.args.Q_thresh_base) * (self.args.Q_thresh_max - self.args.Q_thresh_init)
        thresh_counts = Q_thresh * np.max(counts)

        # scale Q values to the range [0, 1] and set Q values to 0 where Nsa < Q_thresh * max(Nsa)
        Qs = np.array([(float(self.Qsa[(s, a)]) + 1.0) / 2.0 if (s, a) in self.Qsa and self.Nsa[(s, a)] >= thresh_counts else 0 for a in range(self.game.getActionSize())])

        if temp == 0:
            counts = (1 - self.args.Q_factor) * counts + self.args.Q_factor * Qs
            bestAs = np.argwhere(counts == np.max(counts)).flatten()
            bestA = np.random.choice(bestAs)
            probs = np.zeros(self.game.getActionSize())
            probs[bestA] = 1.0
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        probs = [(1 - self.args.Q_factor) * probs[a] + self.args.Q_factor * Qs[a] for a in range(self.game.getActionSize())]
        probs_sum = float(sum(probs))
        probs = [x / probs_sum for x in probs]
        return probs

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        state_stack = []
        v = 0
        num_moves = 0
        winning_node = False

        # find a leaf or terminal node
        while True:
            s = self.game.stringRepresentation(canonicalBoard)

            if s not in self.Es:
                self.Es[s] = self.game.getGameEnded(canonicalBoard, self.player)
            if self.Es[s] != 0:
                # terminal node
                if self.Es[s] == 1:
                    winning_node = True
                v = 1.0
                break

            if s not in self.Ps:
                # leaf node
                self.Ps[s], value = self.nnet.predict(canonicalBoard)
                valids = self.game.getValidMoves(canonicalBoard, 1)
                self.Vs[s] = np.argwhere(valids > 0.0).flatten()
                self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
                sum_Ps_s = np.sum(self.Ps[s])
                if sum_Ps_s > 0:
                    # apply dirichlet noise to the probability distribution
                    self.Ps[s] = (0.75 * self.Ps[s] + 0.25 * valids * np.random.dirichlet(np.full(self.game.getActionSize(), 0.2), 1).flatten())
                    self.Ps[s] /= np.sum(self.Ps[s])  # normalize

                    # enhance the probability for all checking moves where P(s, a) < check_thresh
                    max_Ps_s = np.max(self.Ps[s])
                    enhanced = False
                    for a in self.Vs[s]:
                        if self.Ps[s][a] < self.args.check_thresh and canonicalBoard.gives_check(self.game.decodeAction(a, 1 if canonicalBoard.turn else -1)):
                            self.Ps[s][a] += self.args.check_factor * max_Ps_s
                            enhanced = True
                    if enhanced:
                        self.Ps[s] /= np.sum(self.Ps[s])  # renormalize
                else:
                    # if all valid moves were masked make all valid moves equally probable

                    # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                    # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                    log.error("All valid moves were masked, doing a workaround.")
                    self.Ps[s] = self.Ps[s] + valids
                    self.Ps[s] /= np.sum(self.Ps[s])

                # self.Vs[s] = np.argwhere(valids > 0.0).flatten()
                self.Ns[s] = 0
                v = -value
                break

            valids = self.Vs[s]
            cur_best = -float('inf')
            best_act = -1

            # calculate cpuct and u divisor
            cpuct = math.log((self.Ns[s] + self.args.cpuct_base + 1) / self.args.cpuct_base) + self.args.cpuct_init
            u_divisor = self.args.u_min - math.exp(-self.Ns[s] / self.args.u_base) * (self.args.u_min - self.args.u_init)

            # valid_actions = np.argwhere(valids > 0.0).flatten()
            if len(valids) > 1:
                # pick the action with the highest upper confidence bound
                for a in valids:
                    if (s, a) in self.Qsa:
                        u = self.Qsa[(s, a)] + cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                                u_divisor + self.Nsa[(s, a)])
                    else:
                        u = INITIAL_Q * cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?
                        # u = INITIAL_Q * cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / u_divisor  # Q = 0 ?

                    if u > cur_best:
                        cur_best = u
                        best_act = a

                a = best_act
            else:
                a = valids[0]

            state_stack.append((s, a))

            canonicalBoard, next_player = self.game.getNextState(canonicalBoard, 1, a)
            canonicalBoard = self.game.getCanonicalForm(canonicalBoard, next_player)
            num_moves += 1

        # backpropagate v
        while state_stack:
            s, a = state_stack.pop()
            if (s, a) in self.Qsa:
                self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
                self.Nsa[(s, a)] += 1
            else:
                self.Qsa[(s, a)] = v
                self.Nsa[(s, a)] = 1

            self.Ns[s] += 1
            v = -v
        return winning_node
