from numpy import *
from numpy.random import shuffle
import logging

from AbstractSampler import AbstractSampler

import numpy
numpy.set_printoptions(threshold=10000)


class BaselineSampler(AbstractSampler):
    def __init__(self, arms=[], arg_str=""):
        self.nArms = len(arms)  # Number of arms
        self.lArms = arms       # Arms as a list of arms
        self.iArms = range(self.nArms)   # The indices of the arms
        self.dictArms = dict(zip(self.lArms, self.iArms))  # A dictionary
                                                #taking arms to their indices.
        self.RealWins = zeros([self.nArms, self.nArms])
        self.plays = {}
        for arm1 in self.iArms:
            self.plays[arm1] = {}
            for arm2 in self.iArms:
                if arm1 == arm2:
                    continue
                self.plays[arm1][arm2] = 0
        logging.info("Number of arms = %d" % self.nArms)
        logging.info("Set of arms: %s" % arms)

    def get_arms(self):
        # This returns two arms to compare.
        minplays = None
        mina1 = None
        mina2 = None
        for a1 in self.plays:
            for a2 in self.plays[a1]:
                if minplays is None or self.plays[a1][a2] <= minplays:
                    minplays = self.plays[a1][a2]
                    mina1 = a1
                    mina2 = a2
        self.plays[mina1][mina2] += 1
        logging.info("Selected arm %d and arm %d" % (mina1,
                                                     mina2))
        return self.lArms[mina1], self.lArms[mina2], mina1, mina2

    def update_scores(self, winner, loser, score=1):
        winner = self.dictArms[winner]
        loser = self.dictArms[loser]
        self.RealWins[winner, loser] += score
        logging.info("Score sheet: \n%s" % self.RealWins)
        return winner

    def get_winner(self):
        rWins = self.RealWins
        scores = rWins.sum(axis=1)
        topScore = sort(scores)[-1]
        Inds = nonzero(scores == topScore)[0]
        shuffle(Inds)
        firstPlace = Inds[0]
        return self.lArms[firstPlace]
