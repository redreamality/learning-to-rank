from numpy import *
#set_printoptions(precision=5, suppress=True, linewidth=999999)
#from matplotlib.pyplot import *
from numpy.random import rand, beta, shuffle
import scipy.stats
import logging
import argparse

from AbstractSampler import AbstractSampler


class RelativeThompsonSampler(AbstractSampler):
    def __init__(self, arms=[], arg_str=""):
        parser = argparse.ArgumentParser(prog=self.__class__.__name__)
        parser.add_argument("--sampler", type=str)
        parser.add_argument("--sampler_decay", type=float, default=1.)
        parser.add_argument("--sampler_softscoring",
                            action="store_true", default=False)
        args = vars(parser.parse_known_args(arg_str.split())[0])
        self.nArms = len(arms)  # Number of arms
        self.lArms = arms       # Arms as a list of arms
        self.iArms = range(self.nArms)   # The indices of the arms
        self.dictArms = dict(zip(self.lArms, self.iArms))  # A dictionary
                                                #taking arms to their indices.
        self.RealWins = zeros([self.nArms, self.nArms])
        self.SampleWins = zeros([self.nArms, self.nArms])
        logging.info("Number of arms = %d" % self.nArms)
        logging.info("Set of arms: %s" % arms)
        self.decayFactor = args["sampler_decay"]
        self.softScoring = args["sampler_softscoring"]

    def sample_match(self, ai, aj, matchType='Det'):
        # ai, aj elements of erlf.Arms
        rWins = self.RealWins
        a = rWins[ai, aj] + 2
        b = rWins[aj, ai] + 2
        p = beta(a, b)
        if matchType == 'Rand':
            if rand() < p:
                return ai, aj, 1.  # ie ai beat aj
            else:
                return aj, ai, 1.  # ie aj beat ai
        if matchType == 'Det':
            if p >= 0.5:
                return ai, aj, 1.
            else:
                return aj, ai, 1.
        if matchType == 'SoftWins':
            if ai == aj:
                return ai, aj, 0.
            else:
                return ai, aj, p - 0.5

    def sample_tournament(self):
        # firstPlace, secondPlace, SampleWins = sample_tournament()
        arms = self.iArms
        nA = len(arms)
        self.SampleWins = zeros([nA, nA])
        for ai in arms:
            for aj in arms:
                winner, loser, score = self.sample_match(ai, aj)
                self.SampleWins[winner, loser] += score
        scores = self.SampleWins.sum(axis=1)
        topScores = sort(scores)[-2:].flatten()
        Inds1 = nonzero(scores == topScores[-1])[0]
        if sum(Inds1.shape) == 1:
            Inds2 = nonzero(scores == topScores[0])[0]
            shuffle(Inds2)
            return [Inds1[0], Inds2[0], self.SampleWins]
        else:
            shuffle(Inds1)
            return Inds1[:2].tolist() + [self.SampleWins]

    def do_ts_rel_champ(self, champ, withFig=False):
        arms = self.iArms
        rWins = self.RealWins
        nA = len(arms)
        samples = zeros(nA)
        for ind in range(nA):
            arm = arms[ind]
            if arm == champ:
                samples[ind] = 0.5
                continue
            a = rWins[arm, champ] + 2
            b = rWins[champ, arm] + 2
            samples[ind] = beta(a, b)
#        if withFig:
#            self.plot_arm_densities(champ, 0.95, samples)
        challenger = arms[samples.argmax()]
        return champ, challenger

#    def plot_arm_densities(self, champ, percentile=0.95, samples=array([])):
#        arms = self.iArms
#        rWins = self.RealWins
#        nA = len(arms)
#        champ = self.dictArms[champ]
#        tails = zeros(nA)
#        ucb = zeros(nA)
#        x = linspace(0, 1, 1000)
#        fig = figure(97, figsize=(16, 4.5))
#        clf()
#        for ind in range(nA):
#            arm = arms[ind]
#            if arm == champ:
#                continue
#            a = rWins[arm, champ] + 2
#            b = rWins[champ, arm] + 2
#            tails[ind] = 1 - scipy.stats.beta.cdf(.5, a, b)
#            # 1-beta.cdf(.5,a,b) is the amount of mass you have on the wrong
#            # side of the scale.
#            ucb[ind] = scipy.stats.beta.ppf(percentile, a, b)
#            ax = fig.add_subplot(100 + 10 * nA + ind + 1)
#            prob = scipy.stats.beta.cdf(x, a, b)
#            ax.plot(prob, x)
#            ax.plot([0, 1], [ucb[ind], ucb[ind]])
#            ax.plot([0, 1], [0.5, 0.5], 'r--')
#            if samples.shape[0] > 0:
#                ax.plot(0.5, samples[ind], '*r')
#            ax.set_title('p(prob(a' + str(arm) + ' > a' + str(champ) + '))')
#            grid('on')
#            draw()
#            show()

    def get_arms(self):
        # This returns two arms to compare.
        firstPlace, secondPlace, self.SampleWins = self.sample_tournament()
        firstPlace, secondPlace = self.do_ts_rel_champ(firstPlace)
        logging.info("Selected arm %d and arm %d" % (firstPlace, secondPlace))
        return self.lArms[firstPlace], self.lArms[secondPlace], firstPlace, secondPlace

    def update_scores(self, winner, loser, score=1):
        if self.softScoring == False or winner == loser:
            score = 1
        winner = self.dictArms[winner]
        loser = self.dictArms[loser]
        self.RealWins = self.RealWins * self.decayFactor
        self.RealWins[winner, loser] = self.RealWins[winner, loser] + score
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
