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
from AbstractUserModel import AbstractUserModel


class FederatedClickModel(AbstractUserModel):
    def __init__(self, arg_str):
        self.parmh = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.3, 0.25, 0.2, 0.15]
        self.parmphi = [.68, .61, .48, .34, .28, .2, .11, .1, .08, .06]

    def h(self, i):
        if i < len(self.parmh):
            return self.parmh[i]
        return 0.0

    def p(self, i):
        if i < len(self.parmphi):
            return self.parmphi[i]
        return 0.0

    def b(self, i):
        if i == 0:
            return 1.0
        return float(1) / ((abs(i) + .1))

    def get_clicks(self, result_list, labels):
        """simulate clicks on list l"""
        hposs = [pos for pos, d in enumerate(result_list) if d.get_type()]
        if len(hposs):
            A = np.random.binomial(1, self.h(min(hposs)))
        else:
            A = 0
        c = np.zeros(len(result_list), dtype='int')
        for pos, r in enumerate(result_list):
            label = labels[r[0]]
            if label == 0:
                continue
            e = self.p(pos)
            if A == 1:
                nearest = sorted([(abs(hpos - pos), hpos - pos)
                                   for hpos in hposs])[0][1]
                beta = self.b(nearest)
                e = e + (1 - e) * beta
            E = np.random.binomial(1, e)
            if E and label > 0:
                c[pos] = 1
        return c

    def get_examination_prob(self, result_list):
        hposs = [pos for pos, d in enumerate(result_list) if d.get_type()]
        if len(hposs):
            a = self.h(min(hposs))
        else:
            a = 0.0
        e = np.zeros(len(result_list), dtype='float')
        for pos, _ in enumerate(result_list):
            phi = self.p(pos)
            if len(hposs):
                nearest = sorted([(abs(hpos - pos), hpos - pos)
                                  for hpos in hposs])[0][1]
                beta = self.b(nearest)
            else:
                beta = 0.0
            e[pos] = (a * (phi + (1 - phi) * beta)) + ((1 - a) * phi)
        return e

