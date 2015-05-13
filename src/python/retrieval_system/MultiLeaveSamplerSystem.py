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

from BaselineSamplerSystem import BaselineSamplerSystem


class MultiLeaveSamplerSystem(BaselineSamplerSystem):
    def get_ranked_list(self, query):
        (l, context) = self.comparison.interleave(self.rankers,
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

        per_q = np.zeros([len(self.rankers), len(self.rankers)])

        for i1 in range(len(self.rankers)):
            r1 = self.rankers[i1]
            for i2 in range(len(self.rankers)):
                r2 = self.rankers[i2]
                if outcome[i1] > outcome[i2]:
                    self.sampler.update_scores(r1, r2)
                    per_q[i1, i2] += 1
                if outcome[i1] < outcome[i2]:
                    self.sampler.update_scores(r2, r1)
                    per_q[i2, i1] += 1
        self.iteration += 1
        per_q /= 2
        return self.get_solution(), per_q
