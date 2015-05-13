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
from numpy import *

from BaselineSamplerSystem import BaselineSamplerSystem


class PairedMultiLeaveSamplerSystem(BaselineSamplerSystem):
    def get_ranked_list(self, query):
        self.r1, self.r2, self.i1, self.i2 = self.sampler.get_arms()
        (l, context) = self.comparison.interleave([self.r1, self.r2],
                                                  query,
                                                  self.nr_results)
        self.current_l = l
        self.current_context = context
        self.current_query = query
        return l

    def update_solution(self, clicks):
        outcome = self.comparison.infer_outcome(self.current_l,
                                                self.current_context,
                                                clicks,
                                                self.current_query)
        per_q = zeros([len(self.rankers), len(self.rankers)])

        if outcome[0] >= outcome[1]:
            self.sampler.update_scores(self.r1, self.r2)
            per_q[self.i1, self.i2] += 1
        if outcome[0] <= outcome[1]:
            self.sampler.update_scores(self.r2, self.r1)
            per_q[self.i2, self.i1] += 1
        self.iteration += 1
        return self.get_solution(), per_q
