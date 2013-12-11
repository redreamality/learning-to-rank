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

#    def update_solution(self, clicks):
#        outcome, isw = self.comparison.infer_outcome(self.current_l,
#                                                self.current_context,
#                                                clicks,
#                                                self.current_query)
#        if outcome <= 0:
#            self.sampler.update_scores(self.r1, self.r2, score=1, play=1)
#        else:
#            self.sampler.update_scores(self.r2, self.r1, score=1, play=1)
#
#        for i in range(len(self.sampler.lArms)):
#            for j in range(len(self.sampler.lArms)):
#                a1 = self.sampler.lArms[i]
#                a2 = self.sampler.lArms[j]
#                candidate_context = ([], a1, a2)
#                outcome, isw = self.comparison.infer_outcome(self.current_l,
#                                                            candidate_context,
#                                                            clicks,
#                                                            self.current_query)
#                if outcome <= 0:
#                    self.sampler.update_scores(self.r1, self.r2,
#                                               score=1 * isw, play=isw)
#                else:
#                    self.sampler.update_scores(self.r2, self.r1,
#                                               score=1 * isw, play=isw)
#
#        self.iteration += 1
#        return self.get_solution()
    def update_solution(self, clicks):
        outcome = self.comparison.infer_outcome(self.current_l,
                                                self.current_context,
                                                clicks,
                                                self.current_query)

        for i1 in range(len(self.rankers)):
            r1 = self.rankers[i1]
            for i2 in range(len(self.rankers)):
                r2 = self.rankers[i2]
                if outcome[i1] > outcome[i2]:
                    self.sampler.update_scores(r1, r2)
                else:
                    self.sampler.update_scores(r2, r1)
        self.iteration += 1
        return self.get_solution()
