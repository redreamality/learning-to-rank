from numpy import *
from numpy.random import shuffle
from random import randint
import logging
import argparse
import os
import os.path
import time

from AbstractSampler import AbstractSampler

def getArgmax(A):
    # A is assumed to be a 1D array
    topInds = nonzero(A==A.max())[0]
    return topInds[randint(0,topInds.shape[0]-1)]


class RelativeUCBSampler(AbstractSampler):
    def __init__(self, arms=[], arg_str="", run_count=""):
        parser = argparse.ArgumentParser(prog=self.__class__.__name__)
        parser.add_argument("--sampler", type=str)
        parser.add_argument("--RUCB_alpha_parameter", type=float, default=0.5)
        parser.add_argument("--continue_sampling_experiment", type=str, 
                          default="No")
        parser.add_argument("--old_output_dir", type=str, default="")
        parser.add_argument("--old_output_prefix", type=str, default="")
        args = vars(parser.parse_known_args(arg_str.split())[0])
        self.nArms = len(arms)  # Number of arms
        self.lArms = arms       # Arms as a list of arms
        self.iArms = range(self.nArms)   # The indices of the arms
        self.dictArms = dict(zip(self.lArms, self.iArms))  # A dictionary
                                                #taking arms to their indices.
        self.RealWins = ones([self.nArms, self.nArms])
        self.numPlays = 2*ones([self.nArms, self.nArms])
        self.PMat = self.RealWins / self.numPlays
        self.invSqrtNumPlays = 1./sqrt(self.numPlays)
        logging.info("Number of arms = %d" % self.nArms)
        logging.info("Set of arms: %s" % arms)
        self.t = 1.
        self.UCB = ones([self.nArms,self.nArms])
        self.chatty = False
        self.alpha = args["RUCB_alpha_parameter"]
        if run_count == "":
            self.runMessage = ""
        else:
            self.runMessage = "Run %s: " % str(run_count)
        old_output_dir = args["old_output_dir"]
        old_output_prefix = args["old_output_prefix"]
        if args["continue_sampling_experiment"] == "Yes" and \
                old_output_dir != "" and old_output_prefix != "":
            old_file = os.path.join(old_output_dir, "%s-%d.npz" \
                                    % (old_output_prefix,int(run_count)))
            data = load(old_file)
            time.sleep(int(run_count))
            self.t = data['time']+1
            self.RealWins = data['RealWins']
            self.numPlays = self.RealWins + self.RealWins.T
            self.PMat = self.RealWins / self.numPlays
            self.invSqrtNumPlays = 1./sqrt(self.numPlays)
            data.close()
            logging.info("Done reading "+old_file)

    
    
    def get_UCB(self):
        self.UCB = self.PMat + \
                sqrt(self.alpha*log(self.t)) * self.invSqrtNumPlays
        fill_diagonal(self.UCB,.5)
    
    
    def sample_tournament(self):
        # firstPlace, secondPlace = sample_tournament()
        self.get_UCB()
        potentialChamps = (self.UCB >= .5).all(axis=1)
        if self.t % 1000 == 0:
            logging.info("%s%d- Number of potential champions: %d   "\
                         % (self.runMessage, self.t, potentialChamps.sum()) \
                         +"Potential champions: %s" \
                         % nonzero(potentialChamps)[0].tolist())
        return getArgmax(potentialChamps) # * UCB.max(axis=1))
    
    
    def do_UCB_rel_champ(self,champ):
        arms = self.iArms
        rWins = self.RealWins
        nA = len(arms)
        ucb = self.UCB[:,champ]
        challenger = arms[getArgmax(ucb)]
        return challenger
    
    
    def get_arms(self):
        # This returns two arms to compare.
        firstPlace = self.sample_tournament()
        secondPlace = self.do_UCB_rel_champ(firstPlace)
        if self.chatty and self.t % 1000 == 0:
            logging.info("Iteration %d: Selected arm %d and arm %d \nScore sheet: \n%s" \
                    % (self.t, firstPlace, secondPlace, self.RealWins))
        return self.lArms[firstPlace], self.lArms[secondPlace], firstPlace, secondPlace    
    
    def update_scores(self,winner,loser):
        winner = self.dictArms[winner]
        loser = self.dictArms[loser]
        self.RealWins[winner,loser] += 1
        self.numPlays[winner,loser] += 1
        self.invSqrtNumPlays[winner,loser] = \
                                1./sqrt(self.numPlays[winner,loser])
        self.PMat[winner,loser] = \
            self.RealWins[winner,loser]/self.numPlays[winner,loser]
        self.numPlays[loser,winner] += 1
        self.invSqrtNumPlays[loser,winner] = \
                                1./sqrt(self.numPlays[loser,winner])
        self.PMat[loser,winner] = \
            self.RealWins[loser,winner]/self.numPlays[loser,winner]
        self.t = self.t + 1
        return winner
    
    def get_winner(self):
        rWins = self.RealWins
        scores = rWins.sum(axis=1)
        topScore = sort(scores)[-1]
        Inds = nonzero(scores == topScore)[0]
        shuffle(Inds)
        firstPlace = Inds[0]
        return self.lArms[firstPlace]
    