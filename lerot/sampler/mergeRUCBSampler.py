from numpy import *
import random
import logging
import argparse
from random import randint
import os
import os.path
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


class armTree:
    def __init__(self,iArms,batch_size=4):
        self.batch_size = int(batch_size)
        self.armGroups = []
        nAG = int(ceil(float(len(iArms))/self.batch_size))
        Inds = batch_size * arange(nAG+1)
        Inds[-1] = len(iArms)
        for i in range(len(Inds)-1):
            self.armGroups.append([iArms[j] for j in range(Inds[i],Inds[i+1])])
    
    def pruneGroup(self,i,UCB):
        group = self.armGroups[i]
        if len(group) != UCB.shape[0]:
            logging.info("ERROR: The size of the batch and the dimensions of "+\
                         "UCB matrix do NOT match up. Batch = %s and matrix "+\
                         "is  = %s" % (group,UCB))
        L,W = nonzero(UCB < 0.5)
        for ind in range(L.shape[0]):
            self.armGroups[i].pop(L[ind])
        return L.shape[0] > 0
    
    def mergeGroups(self):
        oldAG = self.armGroups[:]
        random.shuffle(oldAG)
        oldAG.sort(cmp=lambda x,y: cmp(len(x), len(y)))
        self.armGroups = []
        i = 0; j = len(oldAG)-1
        while i <= j:
            if i == j:
                self.armGroups.append(oldAG[i])
                break
            elif len(oldAG[i]) + len(oldAG[j]) > self.batch_size * 1.5:
                self.armGroups.append(oldAG[j])
                j = j-1
            else:
                self.armGroups.append(oldAG[i]+oldAG[j])
                i = i+1; j = j-1
    
    def mergePairOfBatches(self,i,j):
        self.armGroups[i] = self.armGroups[i] + self.armGroups.pop(j)
    
    def numArms(self):
        return sum([len(ag) for ag in self.armGroups])
    
    def __getitem__(self,key):
        return self.armGroups[key % len(self.armGroups)]
    
    def __len__(self):
        return len(self.armGroups)
    
    def index(self,batch):
        return self.armGroups.index(batch)




class mergeRUCBSampler(AbstractSampler):
    def __init__(self, arms=[], arg_str="", run_count=""):
        parser = argparse.ArgumentParser(prog=self.__class__.__name__)
        parser.add_argument("--sampler", type=str)
        parser.add_argument("--RUCB_alpha_parameter", type=float, default=0.5)
        parser.add_argument("--mergeRUCB_batch_size", type=int, default=4)
        parser.add_argument("--mergeRUCB_delta", type=float, default=0.01)
        parser.add_argument("--continue_sampling_experiment", type=str, 
                            default="No")
        parser.add_argument("--old_output_dir", type=str, default="")
        parser.add_argument("--old_output_prefix", type=str, default="")
        args = vars(parser.parse_known_args(arg_str.split())[0])
        self.nArms = len(arms)  # Number of arms
        self.initArms = arms[:]
        self.lArms = arms       # Arms as a list of arms
        if args["continue_sampling_experiment"] != "Yes":
            random.shuffle(self.lArms)
        self.iArms = range(self.nArms)   # The indices of the arms
        self.dictArms = dict(zip(self.lArms,self.iArms))
                                # A dictionary taking arms to their indices.
        self.RealWins = ones([self.nArms, self.nArms])
        self.numPlays = 2*ones([self.nArms, self.nArms])
        self.PMat = self.RealWins / self.numPlays
        self.invSqrtNumPlays = 1./sqrt(self.numPlays)
        logging.info("Number of arms = %d" % self.nArms)
        logging.info("Set of arms: %s" % arms)
        self.alpha = args["RUCB_alpha_parameter"]
        self.batchSize = args["mergeRUCB_batch_size"]
        self.delta = args["mergeRUCB_delta"] # Prob of failure
        self.tArms = armTree(self.iArms,self.batchSize)
        self.UCB = ones([self.batchSize,self.batchSize])
        self.currentBatch = 0
        self.iteration = 1
        self.C = (((4*self.alpha-1)*(self.nArms**2)) /
                        ((2*self.alpha-1)*self.delta))**(1/(2*self.alpha-1))
        self.t = ceil(self.C)+1
        self.chatty = False
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
                self.t = data['time']+ceil(self.C)+1
#                print "[self.lArms.index(a) for a in self.initArms] = ", [self.lArms.index(a) for a in self.initArms]
                Inds = [self.initArms.index(a) for a in self.lArms]
#                print "[self.initArms.index(a) for a in self.lArms] = ", Inds
                self.RealWins = data['RealWins'][ix_(Inds,Inds)]
                self.numPlays = self.RealWins + self.RealWins.T
                self.PMat = self.RealWins / self.numPlays
                self.invSqrtNumPlays = 1./sqrt(self.numPlays)
#                print "data['armGroups'] = ", data['armGroups']
#                print "data['RealWins'] = \n", data['RealWins']
                self.tArms.armGroups = [[self.lArms.index(self.initArms[a])
                                         for a in ag] 
                                        for ag in data['armGroups'].tolist()]
#                print self.tArms.armGroups
                self.iteration = int(data['iteration'])
                self.currentBatch = int(self.t-ceil(self.C)) % len(self.tArms)
                data.close()
                logging.info("Done reading "+old_file)
#                print "RealWins = \n", self.RealWins

    
    
    def getUCB(self):
        Inds = self.tArms[self.currentBatch]
        while len(Inds) <= 1 and len(self.tArms) > 1:
            self.currentBatch = (self.currentBatch+1) % len(self.tArms)
            Inds = self.tArms[self.currentBatch]
        self.UCB = self.PMat[ix_(Inds,Inds)] + \
            sqrt(self.alpha*log(self.t)) * self.invSqrtNumPlays[ix_(Inds,Inds)]
        fill_diagonal(self.UCB,.5)
    
    
    def sampleTournament(self,withFig=False):
        self.getUCB()
        while self.tArms.pruneGroup(self.currentBatch, self.UCB):
            self.getUCB()
        arms = self.tArms[self.currentBatch]
        champ = arms[randint(0,len(arms)-1)]
        return champ
    
    
    def doUCBRelChamp(self,champ,withFig=False):
        champInd = self.tArms[self.currentBatch].index(champ)
        ucb = self.UCB[:,champInd]
        ucb[champInd] = 0
        challengerInd = myArgmax(ucb)
        challenger = self.tArms[self.currentBatch][challengerInd]
        return challenger
    
    
    def get_arms(self,withFig=False):
    # This returns two arms to compare.
        firstPlace = self.sampleTournament(withFig)
        secondPlace = self.doUCBRelChamp(firstPlace)
        r1 = self.lArms[firstPlace]
        r2 = self.lArms[secondPlace]
        i1 = self.initArms.index(r1)
        i2 = self.initArms.index(r2)
        return r1, r2, i1, i2
    
    
    def update_scores(self,r_winner,r_loser):
    # This method can be used to update the scores.
        winner = self.dictArms[r_winner]
        loser = self.dictArms[r_loser]
        if (self.t - ceil(self.C)) % 1000 == 0.:
            ArmGroups = [sorted([self.initArms.index(self.lArms[i]) \
                                            for i in ag]) \
                                            for ag in self.tArms.armGroups]
            logging.info("%s%d- Number of surviving arms: %d  "\
                         % (self.runMessage, self.t - ceil(self.C),
                            sum([len(ag) for ag in ArmGroups]))+\
                        "Surviving groups of arms: %s   " \
                        % ArmGroups)
            W = self.RealWins
            UCB = W/(W+W.T) + sqrt(self.alpha*log(self.t)/(W+W.T))
            Inds = [self.lArms.index(a) for a in self.initArms]
#            print "AG = ", self.tArms.armGroups, "\n W = \n", W[ix_(Inds,Inds)], "\n(alpha,t) = ", (self.alpha,self.t) , "\n UCB = \n", UCB[ix_(Inds,Inds)] * (W[ix_(Inds,Inds)] > 1)
        if self.tArms.numArms() <= self.nArms/(2**self.iteration)+1 \
                            and len(self.tArms) > 1:
            self.tArms.mergeGroups()
            if min([len(a) for a in self.tArms.armGroups])<=0.5*self.batchSize:
                self.tArms.mergeGroups()
            self.iteration = self.iteration + 1
            logging.info("%s%d- Iteration %d" \
                         % (self.runMessage, self.t - ceil(self.C), 
                            self.iteration))
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
        self.currentBatch = (self.currentBatch+1) % len(self.tArms)
        self.t = self.t + 1
        return self.initArms.index(r_winner)
    
    
    def get_winner(self):
    # This method can be called to find out which arm is the best so far.
        self.numPlays = self.RealWins+self.RealWins.T
        PMat = self.RealWins / self.numPlays
        self.champ = myArgmax((PMat > 0.5).sum(axis=1))
        logging.info("mergeRUCBSampler.get_winner() was called!")
        return self.lArms[self.champ]














#################### OLD CODE ########################

###### V3: 
#    def isReady(self,UCB,width):
#        sideUCB = sign(UCB+eye(UCB.shape[0])-0.5)
#        LCB = 1-UCB.T
#        sideLCB = sign(LCB+eye(UCB.shape[0])-0.5)
#        isClear = ((sideUCB * sideLCB) > 0).all(axis=1).any()
#        isClear = isClear & (UCB-LCB < width).all()
#        return isClear
#    
#    def readyBatches(self,UCB,width):
#        ready = zeros(len(self.armGroups))
#        for ind in range(len(self.armGroups)):
#            group = self.armGroups[ind]
#            ucb = UCB[ix_(group,group)]
#            ready[ind] = self.isReady(ucb,width)
#        return ready
#
#    def getLosers(self,UCB):
#        Losers = []
#        for ag in self.armGroups:
#            ucb = UCB[ix_(ag,ag)]
#            losers = [ag[ind] for ind in nonzero((ucb < 0.5).any(axis=1))[0]]
#            Losers = Losers + losers
#        return Losers


#    def getFullUCB(self):
#        rWins = self.RealWins
#        A = rWins
#        B = rWins.T
#        N = maximum(A+B,ones(A.shape))
#        UCB = A/N + sqrt(self.alpha*log(self.t)/N)
#        fill_diagonal(UCB,.5)
#        return UCB
#
#    def sampleTournament(self,withFig=False):
#        self.getUCB()
#        wins = (self.UCB >= .5).sum(axis=1)
#        champInd = myArgmax(potentialChamps)
#        champ = self.tArms[self.currentBatch][champInd]
#        return champ # * UCB.max(axis=1))
#
#
#    def update_scores(self,r_winner,r_loser):
#        # This method can be used to update the scores.
#        winner = self.dictArms[r_winner]
#        loser = self.dictArms[r_loser]
#        self.RealWins[winner,loser] = self.RealWins[winner,loser] + 1
#        if self.t % max(self.nArms,1000) == 0:
#            fullUCB = self.getFullUCB()
#            Losers = self.tArms.getLosers(fullUCB)
#            lWinners = [self.lArms[ind] for ind in set(self.iArms)-set(Losers)]
#            logging.info("%s%d- Number of surviving arms: %d  "\
#                         % (self.runMessage, self.t,
#                            self.nArms - len(Losers))+\
#                         "Surviving arms: %s" \
#                         % sorted([self.initArms.index(a) for a in lWinners]))
#            readyBatches = self.tArms.readyBatches(fullUCB,self.width1)
#            if readyBatches.sum() > 0.75*len(self.tArms):
#                self.tArms.mergeGroups()
#                self.iteration = self.iteration + 1
#                logging.info("%s%d- Iteration %d" \
#                             % (self.runMessage, self.t, self.iteration))
#        self.RealWins[winner,loser] = self.RealWins[winner,loser] + 1
#        self.numPlays[winner,loser] = self.numPlays[winner,loser] + 1
#        self.invSqrtNumPlays[winner,loser] = \
#            1./sqrt(self.numPlays[winner,loser])
#        self.PMat[winner,loser] = \
#            self.RealWins[winner,loser]/self.numPlays[winner,loser]
#        self.numPlays[loser,winner] = self.numPlays[loser,winner] + 1
#        self.invSqrtNumPlays[loser,winner] = \
#            1./sqrt(self.numPlays[loser,winner])
#        self.PMat[loser,winner] = \
#        self.RealWins[loser,winner]/self.numPlays[loser,winner]
#        self.currentBatch = (self.currentBatch+1) % len(self.tArms)
#        self.t = self.t + 1
#        return self.initArms.index(r_winner)



###### V2: would skip over batches that had a clear winner

#    def getUCB(self):
#        keepLooking = True
#        while keepLooking:
#            Inds = self.tArms[self.currentBatch % len(self.tArms)]
#            tempUCB = self.PMat[ix_(Inds,Inds)] + \
#          sqrt(self.alpha*log(self.t)) * self.invSqrtNumPlays[ix_(Inds,Inds)]
#            fill_diagonal(tempUCB,.5)
#            keepLooking = self.tArms.isReady(tempUCB,self.width2)
#            self.currentBatch = (self.currentBatch+1) % len(self.tArms)
#        self.currentBatch = (self.currentBatch-1) % len(self.tArms)
#        self.UCB = tempUCB


###### V1: merging would happen when enough arms were defeated. ##########
#    def mergeGroups(self):
#        oldAG = self.armGroups[:]
#        self.armGroups = []
#        for i in range(len(oldAG)/2):
#            self.armGroups.append(oldAG[2*i]+oldAG[2*i+1])
#        if mod(len(oldAG),2) == 1:
#            self.armGroups.append(oldAG[-1])
#
#
#    def getUCB(self):
#        Inds = self.tArms[self.currentBatch % len(self.tArms)]
#        self.UCB = self.PMat[ix_(Inds,Inds)] + \
#            sqrt(self.alpha*log(self.t)) * self.invSqrtNumPlays[ix_(Inds,Inds)]
#        fill_diagonal(self.UCB,.5)
#    
#    
#    def getFullUCB(self):
#        rWins = self.RealWins
#        A = rWins
#        B = rWins.T
#        N = maximum(A+B,ones(A.shape))
#        UCB = A/N + sqrt(self.alpha*log(self.t)/N)
#        fill_diagonal(UCB,.5)
#        return UCB
#    
#    def sampleTournament(self,withFig=False):
#        self.getUCB()
#        potentialChamps = (self.UCB >= .5).all(axis=1)
#        champInd = myArgmax(potentialChamps)
#        champ = self.tArms[self.currentBatch][champInd]
#        return champ # * UCB.max(axis=1))
#    
#    
#    def doUCBRelChamp(self,champ,withFig=False):
#        champInd = self.tArms[self.currentBatch].index(champ)
#        ucb = self.UCB[:,champInd]
#        if len(self.tArms) > 1:
#            ucb[champInd] = 0 
#        challengerInd = myArgmax(ucb)
#        challenger = self.tArms[self.currentBatch][challengerInd]
#        return challenger
#    
#    
#    def get_arms(self,withFig=False):
#        # This returns two arms to compare.
#        firstPlace = self.sampleTournament(withFig)
#        secondPlace = self.doUCBRelChamp(firstPlace)
#        r1 = self.lArms[firstPlace]
#        r2 = self.lArms[secondPlace]
#        i1 = self.initArms.index(r1)
#        i2 = self.initArms.index(r2)
#        return r1, r2, i1, i2
#
#
#    def update_scores(self,r_winner,r_loser):
#        # This method can be used to update the scores.
#        winner = self.dictArms[r_winner]
#        loser = self.dictArms[r_loser]
#        self.RealWins[winner,loser] = self.RealWins[winner,loser] + 1
#        if self.t % max(self.nArms,1000) == 0:
#            Losers = self.tArms.getLosers(self.getFullUCB())
#            lWinners = [self.lArms[ind] for ind in set(self.iArms)-set(Losers)]
#            logging.info("%s%d- Number of surviving arms: %d  "\
#                         % (self.runMessage, self.t,
#                            self.nArms - len(Losers))+\
#                         "Surviving arms: %s" \
#                         % sorted([self.initArms.index(a) for a in lWinners]))
#            nPotentialChamps = self.nArms - len(Losers)
#            if nPotentialChamps < 1.5*len(self.tArms):
#                self.tArms.mergeGroups()
#                self.iteration = self.iteration + 1
#                logging.info("%s%d- Iteration %d" \
#                             % (self.runMessage, self.t, self.iteration))
#        self.currentBatch = (self.currentBatch+1) % len(self.tArms)
#        self.RealWins[winner,loser] = self.RealWins[winner,loser] + 1
#        self.numPlays[winner,loser] = self.numPlays[winner,loser] + 1
#        self.invSqrtNumPlays[winner,loser] = \
#            1./sqrt(self.numPlays[winner,loser])
#        self.PMat[winner,loser] = \
#            self.RealWins[winner,loser]/self.numPlays[winner,loser]
#        self.numPlays[loser,winner] = self.numPlays[loser,winner] + 1
#        self.invSqrtNumPlays[loser,winner] = \
#            1./sqrt(self.numPlays[loser,winner])
#        self.PMat[loser,winner] = \
#        self.RealWins[loser,winner]/self.numPlays[loser,winner]
#        self.t = self.t + 1
#        return self.initArms.index(r_winner)
