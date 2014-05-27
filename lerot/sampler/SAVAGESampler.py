from numpy import *
set_printoptions(precision=5, suppress=True, linewidth=999999)
from numpy.random import rand, beta
from random import randint
import logging
import argparse
import time


from AbstractSampler import AbstractSampler

def myArgmin(A):
    # A is assumed to be a 1D array
    bottomInds = nonzero(A==A.min())[0]
    return bottomInds[randint(0,bottomInds.shape[0]-1)]

def myArgmax(A):
    # A is assumed to be a 1D array
    topInds = nonzero(A==A.max())[0]
    return topInds[randint(0,topInds.shape[0]-1)]

def my2DArgmin(A):
    # A is assumed to be a 2D array
    m = A.min()
    R,C = nonzero(A==m)
    ind = randint(0,R.shape[0]-1)
    return R[ind], C[ind]

def my2DArgmax(A):
    # A is assumed to be a 2D array
    m = A.max()
    R,C = nonzero(A==m)
    ind = randint(0,R.shape[0]-1)
    return R[ind], C[ind]


class SAVAGESampler(AbstractSampler):
    def __init__(self, arms=[], arg_str="", run_count=""):
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
        self.numPlays = 2*ones([self.nArms,self.nArms])
        self.RealWins = ones([self.nArms,self.nArms])
        logging.info("Number of arms = %d" % self.nArms)
        logging.info("Set of arms: %s" % arms)
        self.softScoring = args["sampler_softscoring"]
        self.horizon = args["sampler_horizon"]
        self.delta = 1. / self.horizon
        self.t = 1
        self.PMat = self.RealWins / self.numPlays
        self.C = self.c(self.numPlays)
        self.C_plus_PMat = self.c(self.numPlays) + self.RealWins/self.numPlays
        self.firstPlace = 0
        self.activePairs = triu(ones([self.nArms,self.nArms]),1)
        self.exploit = False
        self.champ = []
        self.chatty = False
        if run_count == "":
            self.runMessage = ""
        else:
            self.runMessage = "Run %s: " % str(run_count)
        
        
    def c(self,N):
        return sqrt(log(0.0+self.nArms*(self.nArms-1)*self.horizon**2)/(2*N))
    
    def indep_test(self):
        I,J = nonzero(self.activePairs)
        uI, = nonzero(self.activePairs.any(axis=1))
        uJ, = nonzero(self.activePairs.any(axis=0))
        indepArms, = nonzero((self.C_plus_PMat > 0.5).sum(axis=1) < self.nArms)
        newIndepArms = set(indepArms) & (set(uI) | set(uJ))
        if self.t % 1000 == 0:
            logging.info("%s%d- Margin to independence: %f" \
                         % ( self.runMessage, self.t,
                             (self.C_plus_PMat-0.5)[I,J].min() ))
        if len(newIndepArms) > 0:
            logging.info("%s%d- " % (self.runMessage,self.t)+\
                         "indepArms = %s \n" % indepArms+\
                         "newIndepArms = %s" % newIndepArms+\
                         "set(I) = %s, set(J) = %s \n" % (uI,uJ))
            if self.chatty:
                logging.info("PMat = \n%s" % PMat)
                logging.info("C = \n%s" % C)
        for i in newIndepArms:
            self.activePairs[i,:] = 0
            self.activePairs[:,i] = 0
    
    def stop_explore(self):
        I,J = nonzero(self.activePairs)
        if sum(I.shape) == 0:
            return True
        U_cop = (self.PMat > 0.5).sum(axis=1)
        bestArm = myArgmax(U_cop)
        if U_cop[bestArm] == self.nArms-1 and \
            (self.PMat[bestArm,:]-self.C[bestArm,:] > 0.5).sum() == self.nArms-1:
            return True
    
    def get_arms(self,withFig=False):
        # This returns two arms to compare.
        if self.exploit:
            if self.champ == []:
                PMat = self.RealWins / self.numPlays
                self.champ = myArgmax((PMat > 0.5).sum(axis=1))
            self.firstPlace = self.champ
            self.secondPlace = self.champ
        else:
            self.firstPlace, self.secondPlace = \
                            my2DArgmin(self.numPlays * self.activePairs \
                            + (self.activePairs == 0)*(self.numPlays.max()+1))
                                    # The second piece is to get the minimum of
                                    # the NON-zero numbers.
        return self.lArms[self.firstPlace], self.lArms[self.secondPlace], \
                    self.firstPlace, self.secondPlace
    
    
    def update_scores(self, winner, loser, score=1):
        if self.softScoring == False:
            score = 1.
        first = self.dictArms[winner]
        second = self.dictArms[loser]
        if self.exploit:
            self.t = self.t + 1
            return self.dictArms[winner]
        self.RealWins[first, second] += score
        self.numPlays[first, second] += 1
        self.numPlays[second, first] += 1
        self.indep_test()
        if self.stop_explore():
            self.exploit = True
            self.numPlays = self.RealWins+self.RealWins.T
            PMat = self.RealWins / self.numPlays
            self.champ = myArgmax((PMat > 0.5).sum(axis=1))
            if (PMat > 0.5).sum(axis=1).max() != self.nArms - 1 and self.chatty:
                logging.info("update_scores: The selected champion does NOT "+\
                            "beat all the other arms according to the current"+\
                            " estimates. It only beats %d of them." \
                            % (PMat > 0.5).sum(axis=1).max())
        self.t += 1
        self.C[first, second] = self.c(self.numPlays[first, second])
        self.PMat[first, second] = \
                self.RealWins[first, second]/self.numPlays[first, second]
        self.C_plus_PMat[first, second] = self.C[first, second] \
                                        + self.PMat[first, second]
        self.C[second, first] = self.c(self.numPlays[second, first])
        self.PMat[second, first] = \
                self.RealWins[second, first]/self.numPlays[second, first]
        self.C_plus_PMat[second, first] = self.C[second, first] \
                                        + self.PMat[second, first]
        return self.dictArms[winner]
    
    def get_winner(self):
        if self.champ == []:
            self.numPlays = self.RealWins+self.RealWins.T
            PMat = self.RealWins / self.numPlays
            self.champ = myArgmax((PMat > 0.5).sum(axis=1))
            if (PMat > 0.5).sum(axis=1).max() != self.nArms - 1 and self.chatty:
                logging.info("get_winner: The selected champion does NOT beat"+\
                             " all the other arms according to the current "+\
                             "estimates. It only beats %d of them." \
                             % (PMat > 0.5).sum(axis=1).max())
        return self.lArms[self.champ]
