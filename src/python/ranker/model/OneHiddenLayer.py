import numpy as np
from AbstractRankingModel import AbstractRankingModel


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
    