from numpy import *
from numpy.random import rand, beta
from random import randint
import scipy.stats
import logging
import argparse

from AbstractSampler import AbstractSampler


def myArgmin(A):
    # A is assumed to be a 1D array
    topInds = nonzero(A==A.min())[0]
    return topInds[randint(0,topInds.shape[0]-1)]

def myArgmax(A):
    # A is assumed to be a 1D array
    topInds = nonzero(A==A.max())[0]
    return topInds[randint(0,topInds.shape[0]-1)]


class fastBeta:
    def __init__(self,W,depth=100):
        # W - a square matrix containing a scoresheet
        self.depth = depth
        self.shape = W.shape
        self.W = maximum(W,ones(W.shape))
        self.allSamples = zeros(self.shape+(depth,))
        self.sampleAllBeta()
        self.depthIndex = 0
        self.upper_or_lower = 'UPPER'
    
    def sampleAllBeta(self):
        self.allSamples = 0.5*ones(self.shape+(self.depth,))
        for r,c in [(row,col) for row in range(self.shape[0]) \
                              for col in range(self.shape[1]) if row != col]:
            self.allSamples[r,c,:] = beta(self.W[r,c],self.W[c,r],self.depth)
    
    def update(self,r,c,w):
        self.W[r,c] = w
        self.allSamples[r,c,:] = beta(self.W[r,c],self.W[c,r],self.depth)
        self.allSamples[c,r,:] = beta(self.W[c,r],self.W[r,c],self.depth)
    
    def getSamples(self):
        if self.upper_or_lower == 'UPPER':
            if self.depthIndex > self.depth-1:
                self.depthIndex = 0
                self.sampleAllBeta()
            self.upper_or_lower = 'LOWER'
            return triu(self.allSamples[:,:,self.depthIndex])
        elif self.upper_or_lower == 'LOWER':
            self.depthIndex = self.depthIndex + 1
            self.upper_or_lower = 'UPPER'
            return tril(self.allSamples[:,:,self.depthIndex-1])





class RelativeThompsonSampler(AbstractSampler):
    def __init__(self, arms=[], arg_str="", run_count=""):
        parser = argparse.ArgumentParser(prog=self.__class__.__name__)
        parser.add_argument("--sampler", type=str)
        args = vars(parser.parse_known_args(arg_str.split())[0])
        self.nArms = len(arms)  # Number of arms
        self.lArms = arms       # Arms as a list of arms
        self.iArms = range(self.nArms)   # The indices of the arms
        self.dictArms = dict(zip(self.lArms, self.iArms))  # A dictionary
                                                #taking arms to their indices.
        self.RealWins = ones([self.nArms, self.nArms])
        self.SampleWins = zeros([self.nArms, self.nArms])
        logging.info("Number of arms = %d" % self.nArms)
        logging.info("Set of arms: %s" % arms)
        self.t=1
        self.betaSamples = fastBeta(self.RealWins)
        self.chatty=False
        if run_count == "":
            self.runMessage = ""
        else:
            self.runMessage = "Run %s: " % str(run_count)


    def sample_tournament(self):
        # firstPlace, secondPlace, SampleWins = sample_tournament()
        Samples = self.betaSamples.getSamples()
        self.SampleWins = (Samples > 0.5) + ((Samples > 0) & (Samples < 0.5)).T
        if self.t % 1000 == 0:
            logging.info("%s%d- Top tounament score = %d    " \
                         % ( self.runMessage, self.t, 
                             self.SampleWins.sum(axis=1).max() ) \
                        + "Top tournament scorers from weak to strong: %s" \
                        % self.SampleWins.sum(axis=1).argsort()[-20:])
        return myArgmax(self.SampleWins.sum(axis=1))


    def do_TS_rel_champ(self, champ):
        samples = zeros(self.nArms)
        for arm in range(self.nArms):
            if arm == champ:
                samples[arm] = 0.5
            else:
                a = self.RealWins[arm, champ]
                b = self.RealWins[champ, arm]
                samples[arm] = beta(a, b)
        challenger = samples.argmax()
        return challenger


    def get_arms(self):
        # This returns two arms to compare.
        firstPlace = self.sample_tournament()
        secondPlace = self.do_TS_rel_champ(firstPlace)
        if self.chatty:
            logging.info("%d- Selected arm %d and arm %d \nScore sheet: \n%s" \
                         % (self.t, firstPlace, secondPlace, self.RealWins))
        return self.lArms[firstPlace], self.lArms[secondPlace], firstPlace, secondPlace


    def update_scores(self, winner, loser):
    # This method can be used to update the scores.
        winner = self.dictArms[winner]
        loser = self.dictArms[loser]
        self.RealWins[winner, loser] += 1
        self.betaSamples.update(winner,loser,self.RealWins[winner, loser])
        self.t += 1
        return winner


    def get_winner(self):
    # This method can be called to find out which arm is the best so far.
        self.numPlays = self.RealWins+self.RealWins.T
        PMat = self.RealWins / self.numPlays
        self.champ = myArgmax((PMat > 0.5).sum(axis=1))
        logging.info("mergeRUCBSampler.get_winner() was called!")
        return self.lArms[self.champ]









##############  OLD  ################

#    def sample_match(self, ai, aj):
#        # ai, aj elements of erlf.Arms
#        rWins = self.RealWins
#        a = rWins[ai, aj]
#        b = rWins[aj, ai]
#        p = beta(a, b)
#        if p >= 0.5:
#            return ai, aj, 1.
#        else:
#            return aj, ai, 1.
