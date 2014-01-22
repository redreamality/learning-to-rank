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
        self.RealWins = ones([self.nArms, self.nArms])
        self.plays = ones([self.nArms, self.nArms]) * 2
        logging.info("Number of arms = %d" % self.nArms)
        logging.info("Set of arms: %s" % arms)

    def get_arms(self):
        # This returns two arms to compare.
        minplays = None
        mina1 = None
        mina2 = None

        for i in self.iArms:
            for j in self.iArms:
                if i == j: 
                    continue
                if minplays is None or self.plays[i, j] <= minplays:
                    minplays = self.plays[i, j]
                    mina1 = i
                    mina2 = j
        return self.lArms[mina1], self.lArms[mina2], mina1, mina2

    def update_scores(self, i1, i2, score=1, play=1):
        a1 = self.dictArms[i1]
        a2 = self.dictArms[i2]
        self.plays[a1, a2] += play
        self.plays[a2, a1] += play
        self.RealWins[a1, a2] += score
        #logging.info("Score sheet: \n%s" % self.RealWins)

    def get_sheet(self):
        return self.RealWins / self.plays

    def get_winner(self):
        rWins = self.RealWins / self.plays
        scores = rWins.sum(axis=1)
        topScore = sort(scores)[-1]
        Inds = nonzero(scores == topScore)[0]
        shuffle(Inds)
        firstPlace = Inds[0]
        return self.lArms[firstPlace]
