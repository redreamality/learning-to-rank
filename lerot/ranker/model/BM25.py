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
from ...utils import sample_unit_sphere


class BM25(AbstractRankingModel):

    def __init__(self, feature_count):
        self.feature_count = 3

    def initialize_weights(self, init_method):
        if init_method == "random":
            return sample_unit_sphere(self.feature_count)
        return np.array([2.5, 0, 0.8])

    def score(self, features, w):
        scores = []
        for docfeatures in features:
            k1, k3, b = w
            s = 0.0
            nr_terms = len(docfeatures) / 4
            for i in range(nr_terms):
                idf = docfeatures[i * 4]
                tf = docfeatures[i * 4 + 1]
                qtf = docfeatures[i * 4 + 2]
                dl = docfeatures[i * 4 + 3]
                if dl == 0:
                    continue
                s += ((idf * tf * (k1 + 1)) / (tf + k1 * (1-b+b*dl))) * (((k3+1)*qtf)/(k3+qtf))
            scores.append(s)
        return scores
    
