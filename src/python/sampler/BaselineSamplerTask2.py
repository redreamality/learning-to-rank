from numpy import *
from numpy.random import shuffle
import logging

from BaselineSampler import BaselineSampler

import numpy
numpy.set_printoptions(threshold=10000)


class BaselineSamplerTask2(BaselineSampler):
    def get_arms(self):
        # This returns two arms to compare.
        minplays = None
        mina1 = None
        mina2 = None

        j = self.production
        for i in self.iArms:
            if i == j:
                continue
            if minplays is None or self.plays[i, j] <= minplays:
                minplays = self.plays[i, j]
                mina1 = i
                mina2 = j
        return self.lArms[mina1], self.lArms[mina2], mina1, mina2
