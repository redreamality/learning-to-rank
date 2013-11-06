from numpy import *
set_printoptions(precision=5, suppress=True, linewidth=999999)
from numpy.random import rand, beta, shuffle
import logging
import argparse

from AbstractSampler import AbstractSampler

def myArgmin(A):
    # A is assumed to be a 1D array
    return nonzero(A == A.min())[0]

def myArgmax(A):
    # A is assumed to be a 1D array
    return nonzero(A == A.max())[0]


class BeatTheMeanSampler(AbstractSampler):
    def __init__(self, arms=[], arg_str=""):
        parser = argparse.ArgumentParser(prog=self.__class__.__name__)
        parser.add_argument("--sampler", type=str)
        parser.add_argument("--sampler_decay", type=float, default=1.)
        parser.add_argument("--sampler_softscoring",
                            action="store_true", default=False)
        # Parameters specific to Beat the Mean:
        parser.add_argument("--sampler_horizon", type=int, default=1000)
        parser.add_argument("--sampler_non_transitivity", 
                            type=float, default=1.)
        args = vars(parser.parse_known_args(arg_str.split())[0])
        self.nArms = len(arms)  # Number of arms
        self.lArms = arms       # Arms as a list of arms
        self.iArms = range(self.nArms)   # The indices of the arms
        self.dictArms = dict(zip(self.lArms, self.iArms))  # A dictionary
        #taking arms to their indices.
        self.numPlays = zeros([self.nArms,self.nArms])
        self.RealWins = zeros([self.nArms,self.nArms])
        logging.info("Number of arms = %d" % self.nArms)
        logging.info("Set of arms: %s" % arms)
        self.decayFactor = args["sampler_decay"]
        self.softScoring = args["sampler_softscoring"]
        self.horizon = args["sampler_horizon"]
        self.gamma = args["sampler_non_transitivity"]
        self.delta = 1. / (2 * self.horizon * self.nArms)
        self.t = 0
        self.firstPlace = 0

    def get_arms(self):
        # This returns two arms to compare.
        Wins = self.RealWins
        N = self.numPlays.sum(axis=1)
        Inds = myArgmin(N);        shuffle(Inds);  self.firstPlace = Inds[0]
        Inds = array(self.iArms);  shuffle(Inds);  secondPlace = Inds[0]
        self.numPlays[self.firstPlace,secondPlace] = \
                self.numPlays[self.firstPlace,secondPlace] + 1
        logging.info("Selected arm %d and arm %d" % \
                     (self.firstPlace, secondPlace))
        return self.lArms[self.firstPlace], self.lArms[secondPlace],self.firstPlace, secondPlace

    def online_beat_the_mean(self):
        Wins = self.RealWins.sum(axis=1)
        N = self.numPlays.sum(axis=1)
        N = N + (N == 0) # This is just to avoid dividing by zero.
        P = Wins/N
        n = N.min()
        c = 3. * self.gamma * sqrt(log(1./self.delta)/n)
        if amin(P+c) <= amax(P-c):
            Inds = myArgmin(P);  shuffle(Inds);  ind = Inds[0]
            newInds = sorted(list(set(range(self.nArms)) - set([ind])))
            self.nArms = self.nArms - 1
            self.lArms = [self.lArms[i] for i in newInds]
            self.iArms = range(self.nArms)
            self.dictArms = dict(zip(self.lArms, self.iArms))
            self.numPlays = self.numPlays[newInds][:,newInds]
            self.RealWins = self.RealWins[newInds][:,newInds]

    def update_scores(self, winner, loser, score=1):
        if self.softScoring == False:
            score = 1.
        if winner == loser:
            score = float(rand() < 0.5)
        self.numPlays = self.numPlays * self.decayFactor
        self.RealWins = self.RealWins * self.decayFactor
        if self.firstPlace == self.dictArms[winner]:
            first = self.dictArms[winner]
            second = self.dictArms[loser]
            self.RealWins[first, second] = self.RealWins[first, second] + score
            self.numPlays[first, second] = self.numPlays[first, second] + 1
        else:
            first = self.dictArms[loser]
            second = self.dictArms[winner]
            self.numPlays[first, second] = self.numPlays[first, second] + 1
        self.online_beat_the_mean()
        logging.info("Score sheet: \n%s" % self.RealWins)
        return self.dictArms[winner]
    
    def get_winner(self):
        Wins = self.RealWins.sum(axis=1)
        N = self.numPlays.sum(axis=1)
        N = N + (N == 0) # This is just to avoid dividing by zero.
        P = Wins/N
        Inds = myArgmax(P)
        shuffle(Inds)
        firstPlace = Inds[0]
        return self.lArms[firstPlace]
