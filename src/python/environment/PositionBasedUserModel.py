import numpy as np
from numpy import zeros

from AbstractUserModel import AbstractUserModel


class PositionBasedUserModel(AbstractUserModel):
    """Defines a positions based user model."""

    def __init__(self, p):
        self.p_param = p

    def p(self, i):
        return self.p_param ** i

    def get_clicks(self, result_list, labels):
        """simulate clicks on list l"""
        c = zeros(len(result_list), dtype='int')
        for pos, d in enumerate(result_list):
            E = np.random.binomial(1, self.p(pos))
            label = labels[d]
            if E and label:
                c[pos] = 1
        return c

    def get_examination_prob(self, result_list):
        return [self.p(i) for i in range(len(result_list))]
