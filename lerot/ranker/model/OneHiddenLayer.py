# This file is part of Lerot.
#
# Lerot is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Lerot is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Lerot.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from .AbstractRankingModel import AbstractRankingModel


class OneHiddenLayer(AbstractRankingModel):

    def __init__(self, feature_count):
        self.hiddensize = 10
        self.inputsize = feature_count + 1
        self.feature_count = (self.inputsize * self.hiddensize) \
                                                            + self.hiddensize

    def initialize_weights(self, init_method):
        return AbstractRankingModel.initialize_weights(self, init_method)

    def score(self, features, w):
        features = np.vstack((np.ones(len(features)),
                              features.transpose())).transpose()
        w1 = w[:-self.hiddensize]
        w1 = w1.reshape((self.inputsize, self.hiddensize))
        w2 = w[-self.hiddensize:]
        s = np.tanh(np.dot(np.tanh(np.dot(features, w1)), w2))
        return s
    
