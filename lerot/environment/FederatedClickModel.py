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

from collections import defaultdict
import itertools
import numpy as np

from AbstractUserModel import AbstractUserModel


class FederatedClickModel(AbstractUserModel):
    def __init__(self, arg_str):
        self.parmh = {'text': [0.95, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
                      'media': [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.3, 0.25, 0.2, 0.15, 0.10, 0.05]}
        self.parmphi = [.68, .61, .48, .34, .28, .2, .11, .1, .08, .06]
        args = arg_str.split()
        self.pargamma = {'text': float(args[0]), 'media': float(args[1])}

    def h(self, i, serp_len, vert):
        return self.getParamRescaled(i, serp_len, self.parmh[self.getVertClass(vert)])

    def p(self, i, serp_len):
        return self.getParamRescaled(i, serp_len, self.parmphi)

    def b(self, i, vert):
        return min(1, 1.0 / ((abs(i) + self.pargamma[self.getVertClass(vert)])))

    @staticmethod
    def getParamRescaled(rank, serp_len, param_vector):
        assert rank < serp_len
        if serp_len <= len(param_vector):
            return param_vector[rank]
        origin_rank = float(rank) / (serp_len - 1) * (len(param_vector) - 1)
        left = int(origin_rank)
        delta = origin_rank - left
        if delta < 0.01:
            return param_vector[left]
        return param_vector[left] * (1 - delta) + param_vector[left + 1] * delta

    @staticmethod
    def getVertClass(vert_type):
        if vert_type in ['Answer', 'Blog', 'Books', 'Discussion',
                         'News', 'Scholar', 'Wiki']:
            return 'text'
        elif vert_type in ['Image', 'Recipe', 'Shopping', 'Video', 'Apps']:
            return 'media'
        else:
            raise NotImplementedError('Unknown vertical type: %s' % vert_type)

    def get_clicks(self, result_list, labels, **kwargs):
        """Simulate clicks on the result_list.
            - labels contain relevance labels indexed by the docid
        """
        N = len(result_list)
        orientation = kwargs.get('orientation')
        if orientation is None:
            orientation = defaultdict(lambda: 1.0)
        vert_types = set(d.get_type() for d in result_list if d.get_type() != 'Web')
        biased_verticals = set([])      # the set of verticals for which A^j == True
        for vert in vert_types:
            hposs = [i for i, d in enumerate(result_list) if d.get_type() == vert]
            A = np.random.binomial(1, self.h(hposs[0], N, vert) * orientation[vert])
            if A:
                biased_verticals.add(vert)
        examination_probs = self._examination_prob(result_list, biased_verticals)
        return [1 if labels[d.get_id()] > 0 and np.random.binomial(1, e) else 0 \
            for (e, d) in zip(examination_probs, result_list)]

    def get_examination_prob(self, result_list, **kwargs):
        N = len(result_list)
        orientation = kwargs.get('orientation')
        if orientation is None:
            orientation = defaultdict(lambda: 1.0)
        vert_types = list(set(d.get_type() for d in result_list if d.get_type() != 'Web'))
        # P(A_j = 1)
        p_A_j_1 = np.zeros(len(vert_types))
        for j, vert in enumerate(vert_types):
            hposs = [i for i, d in enumerate(result_list) if d.get_type() == vert]
            p_A_j_1[j] = self.h(hposs[0], N, vert) * orientation[vert]
        # P(E_i = 1) = \sum_A P(A) \cdot P(E_i = 1 \mid A)
        p_E = np.zeros(N)
        # A is a vector of attractiveness values of length `len(vert_types)`
        for A in itertools.product([0, 1], repeat=len(vert_types)):
            biased_verticals = set(v for (a, v) in zip(A, vert_types) if a)
            # P(A) = \prod_j P(A_j)
            p_A = 1.0
            for j, a in enumerate(A):
                p_A *= p_A_j_1[j] if a else (1 - p_A_j_1[j])
            # P(E = 1 \mid A)
            p_E_1_mid_A = self._examination_prob(result_list, biased_verticals)
            p_E += p_A * np.array(p_E_1_mid_A)
        return p_E

    def _examination_prob(self, result_list, biased_verticals):
        N = len(result_list)
        examination_probs = []
        for pos, d in enumerate(result_list):
            beta = 0
            for vert in biased_verticals:
                nearest = min((i for i, d in enumerate(result_list) \
                               if d.get_type() == vert),
                              key=lambda i: (abs(i - pos), i))
                beta = max(beta, self.b(nearest - pos, vert))
            beta = min(1, beta)
            phi = self.p(pos, N)
            e = phi + (1 - phi) * beta
            examination_probs.append(e)
        return examination_probs
