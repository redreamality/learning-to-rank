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
from numpy import zeros

from AbstractUserModel import AbstractUserModel


class PositionBasedUserModel(AbstractUserModel):
    """Defines a positions based user model."""

    def __init__(self, p):
        self.p_param = p

    def p(self, i):
        return self.p_param ** i

    def get_clicks(self, result_list, labels, **kwargs):
        """simulate clicks on list l"""
        c = zeros(len(result_list), dtype='int')
        for pos, d in enumerate(result_list):
            E = np.random.binomial(1, self.p(pos))
            label = labels[d.get_id()]
            if E and label:
                c[pos] = 1
        return c

    def get_examination_prob(self, result_list, **kwargs):
        return [self.p(i) for i in range(len(result_list))]
