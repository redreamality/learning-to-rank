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

"""
Runs an online evaluation experiment. The "environment" logic is located here,
e.g., sampling queries, observing user clicks, external evaluation of result
lists
"""

from AbstractLearningExperiment import AbstractLearningExperiment
from utils import get_class
import logging
import numpy as np
import sys


class SamplingExperiment(AbstractLearningExperiment):
    def __init__(self, training_queries, test_queries, feature_count, log_fh,
                args):
        AbstractLearningExperiment.__init__(self, training_queries,
                                            test_queries, feature_count,
                                            log_fh, args)

        evalsystemclass = get_class("retrieval_system.BaselineSamplerSystem")
        self.evalsystem = evalsystemclass(self.feature_count, self.system_args)
        self.evalsystem.rankers = self.system.rankers
        evalumargs = self.um_args
        evalum = self.um_class(evalumargs)

        self.query_keys = sorted(self.training_queries.keys())
        self.query_length = len(self.query_keys)

        for query_count in range(10000):
            qid = self._sample_qid(self.query_keys, query_count,
                                   self.query_length)
            query = self.training_queries[qid]
            # get result list for the current query from the system
            result_list = self.evalsystem.get_ranked_list(query)
            # generate click feedback
            clicks = evalum.get_clicks(result_list, query.get_labels())
            # send feedback to system
            self.groundtruth = self.evalsystem.update_solution(clicks)

    def evaluate(self, solution):
        def largerthan05(v):
            return (v - .5) > sys.float_info.epsilon

        score = 0.0
        for (i, j), val in np.ndenumerate(self.groundtruth):
            if largerthan05(val) != largerthan05(solution[i, j]):
                score += 1
        #print solution
        #print self.groundtruth

        score1 = score / (i * j)
        score2 = float(np.sum(np.absolute(np.subtract(self.groundtruth,
                                                    solution))) / (i * j))
        #logging.info("Score: %.3f %.3f" % (score1, score2))
        return score1, score2

    def run(self):
        summary = {"binary_diff": [],
                   "diff": []}
        # process num_queries queries
        for query_count in range(self.num_queries):
            qid = self._sample_qid(self.query_keys, query_count,
                                   self.query_length)
            query = self.training_queries[qid]
            # get result list for the current query from the system
            result_list = self.system.get_ranked_list(query)
            # generate click feedback
            clicks = self.um.get_clicks(result_list, query.get_labels())
            # send feedback to system
            solution = self.system.update_solution(clicks)
            s1, s2 = self.evaluate(solution)
            summary["binary_diff"].append(s1)
            summary["diff"].append(s2)
        return summary
