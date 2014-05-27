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

import math

from collections import defaultdict

from .AbstractEval import AbstractEval

class AsRbpEval(AbstractEval):
    """Compute AS_RBP metric as described in [1].

       [1] Zhou, K. et al. 2012. Evaluating aggregated search pages. SIGIR. (2012).
    """

    def __init__(self, alpha=10, beta=0.8,):
        self.alpha = alpha
        self.beta = beta

        self.effort = defaultdict(lambda: 3.0)
        self.effort['Image'] = 1.0
        self.effort['Video'] = 6.0

    def _transform_orientation(self, orient):
        assert 0 <= orient <= 1
        if orient == 0:
            return 0.0
        elif orient == 1:
            return 1.0
        else:
            return 1 / (1 + self.alpha ** (-math.log10(orient / (1 - orient))))

    def get_value(self, ranking, labels, orientations, cutoff=-1):
        if cutoff == -1:
            cutoff = len(ranking)
        gain = 0.0
        effort = 0.0
        examination = 1.0
        for i, doc in enumerate(ranking[:cutoff]):
            cur_vert_type = doc.get_type()
            if i > 0 and cur_vert_type != ranking[i-1].get_type():
                examination *= self.beta
            gain += (self._transform_orientation(orientations[cur_vert_type]) *
                     labels[doc.get_id()] * examination)
            effort += self.effort[cur_vert_type] * examination
        return gain / (1.0 if effort == 0 else effort)
