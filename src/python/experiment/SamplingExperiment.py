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

        evaluation_class = get_class("evaluation.NdcgEval")
        self.evaluation = evaluation_class()

        rankers = self.system.rankers
        n = len(rankers)
        ndcgs = []
        for r in range(n):
            ndcgs.append(self.evaluation.evaluate_all(rankers[r], test_queries,
                                                 cutoff=self.system.nr_results)
                         )

        wins = np.zeros([n, n])
        plays = np.zeros([n, n])
        self.diff = np.zeros([n, n])

        for r1 in range(n):
            for r2 in range(n):
                if ndcgs[r1] > ndcgs[r2]:
                    wins[r1, r2] += 1
                else:
                    wins[r2, r1] += 1
                self.diff[r1, r2] = 0.5 * (ndcgs[r1] - ndcgs[r2]) + 0.5
                plays[r1, r2] += 1
                plays[r2, r1] += 1
        self.groundtruth = wins / plays

        self.query_keys = sorted(self.training_queries.keys())
        self.query_length = len(self.query_keys)

    def evaluate(self, solution, result_list, query, per_q):
        def largerthan05(v):
            return (v - .5) > sys.float_info.epsilon

        def sgn(v):
            if v < (0 - sys.float_info.epsilon):
                return -1
            elif v > (0 + sys.float_info.epsilon):
                return 1
            else:
                return 0

        size1 = 0
        size2 = 0
        score1 = 0.0
        for (i, j), val in np.ndenumerate(self.groundtruth):
            size1 += 1
            if i == j:
                continue
            if largerthan05(val) != largerthan05(solution[i, j]):
                score1 += 1
            size2 += 1
        #print solution
        #print self.groundtruth

        score1 = score1 / size2
        score2 = float(np.sum(np.absolute(np.subtract(self.groundtruth,
                                                    solution))) / size1)

        score3 = 0.0
        for (i, j), val in np.ndenumerate(self.groundtruth):
            if i == j:
                continue
            if sgn(val - 0.5) != sgn(solution[i, j] - 0.5):
                score3 += 1
        score3 = score3 / size2

        score4 = 0.0
        for (i, j), val in np.ndenumerate(self.groundtruth):
            if i == j:
                continue
            if sgn(val - 0.5) != sgn(solution[i, j] - 0.5):
                score4 += abs(self.diff[i, j] - solution[i, j])
        score4 = score4 / size2

        score5 = self.evaluation.evaluate_ranking(result_list, query)

        score6 = 0.0
        for (i, j), val in np.ndenumerate(self.groundtruth):
            if i == j:
                continue
            score6 += abs(0.5 - solution[i, j])
        score6 = score6 / size2

        score7 = 0.0
        for (i, j), val in np.ndenumerate(self.groundtruth):
            if i == j:
                continue
            score7 += abs(0.5 - per_q[i, j])
        score7 = score7 / size2

        score8 = 0.0
        for (i, j), val in np.ndenumerate(self.groundtruth):
            if i == j:
                continue
            if sgn(val - 0.5) != sgn(per_q[i, j] - 0.5):
                score8 += abs(self.diff[i, j])
        score8 = score8 / size2

        relaxed = 0
        try:
            if self.system.comparison.relaxed:
                relaxed = 1
        except:
            pass

        #logging.info("Score: %.3f %.3f" % (score1, score2))
        return {"binary_diff": float(score1),
                "diff": float(score2),
                "binary_diff_2": float(score3),
                "binary_scaled": float(score4),
                "binary_ndcg": float(score8),
                "online_ndcg": float(score5),
                "bias": float(score6),
                "per_q_bias": float(score7),
                "relaxed": int(relaxed)}

    def run(self):
        summary = {}
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
            solution, per_q = self.system.update_solution(clicks)
            scores = self.evaluate(solution, result_list, query, per_q)
            #print "query", query_count, scores
            for k in scores:
                if not k in summary:
                    summary[k] = []
                summary[k].append(scores[k])
        summary["final_solution"] = [[float(y) for y in list(x)]
                                     for x in list(solution)]
        summary["groundtruth"] = [[float(y) for y in list(x)]
                                     for x in list(self.groundtruth)]

        summary["groundtruthndcg"] = [[float(y) for y in list(x)]
                                     for x in list(self.diff)]
        return summary
