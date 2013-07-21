import numpy as np
from AbstractRankingModel import AbstractRankingModel


class Linear(AbstractRankingModel):

    def score(self, features, w):
        return np.dot(features, w)
