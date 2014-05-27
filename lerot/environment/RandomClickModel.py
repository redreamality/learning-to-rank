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

class RandomClickModel(AbstractUserModel):
    """Defines a positions based user model."""

    def __init__(self, p=0.5):
        self.p = p

    def get_clicks(self, result_list, labels, **kwargs):
        """simulate clicks on list l"""
        c = zeros(len(result_list), dtype='int')
        for pos, d in enumerate(result_list):
            if np.random.binomial(1, self.p):
                c[pos] = 1
        return c

