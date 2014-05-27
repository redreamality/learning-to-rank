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

import unittest
import sys
import os
import cStringIO
import numpy as np

sys.path.insert(0, os.path.abspath('..'))

from lerot import query
from DeterministicRankingFunction import DeterministicRankingFunction
from ProbabilisticRankingFunction import ProbabilisticRankingFunction


class TestRankers(unittest.TestCase):
    def setUp(self):
        self.test_num_features = 6
        test_query = """
        4 qid:1 1:2.6 2:1 3:2.1 4:0 5:2 6:1.4 # highly relevant
        1 qid:1 1:1.2 2:1 3:2.9 4:0 5:2 6:1.9 # bad
        0 qid:1 1:0.5 2:1 3:2.3 4:0 5:2 6:5.6 # not relevant
        0 qid:1 1:0.5 2:1 3:2.3 4:0 5:2 6:5.6 # not relevant
        """

        self.query_fh = cStringIO.StringIO(test_query)
        self.queries = query.Queries(self.query_fh, self.test_num_features)
        self.query = self.queries['1']

        zero_weight_str = "0 0 0 0 0 0"
        self.zero_weights = np.asarray([float(x) for x in
            zero_weight_str.split()])

        weight_str = "0 0 1 0 0 0"
        self.weights = np.asarray([float(x) for x in weight_str.split()])

    def testDeterministicRandomTiesZero(self):
        rf = DeterministicRankingFunction(-1, self.zero_weights)
        test_rankings = {"0,1,2,3": 0, "3,2,1,0": 0, "3,1,0,2": 0}
        trials = 0
        MAX_TRIALS = 1000
        while trials < MAX_TRIALS and 0 in test_rankings.values():
            trials += 1
            rf.init_ranking(self.query)
            observed_ranking = ",".join(str(d) for d in rf.docids)
            if observed_ranking in test_rankings:
                test_rankings[observed_ranking] += 1
        for ranking, count in test_rankings.items():
            self.assertNotEqual(0, count, "Test failed for: %s" % ranking)

    def testDeterministicRandomTies(self):
        rf = DeterministicRankingFunction(-1, self.weights)
        rf.init_ranking(self.query)
        self.assertIn(rf.docids, [[1, 2, 3, 0], [1, 3, 2, 0]])

    def testDeterministicFirstTiesZero(self):
        rf = DeterministicRankingFunction(-1, self.zero_weights, ties="first")
        rf.init_ranking(self.query)
        self.assertEqual(rf.docids, [0, 1, 2, 3])

    def testDeterministicFirstTies(self):
        rf = DeterministicRankingFunction(-1, self.weights, ties="first")
        rf.init_ranking(self.query)
        self.assertEqual(rf.docids, [1, 2, 3, 0])

    def testDeterministicLastTiesZero(self):
        rf = DeterministicRankingFunction(-1, self.zero_weights, ties="last")
        rf.init_ranking(self.query)
        self.assertEqual(rf.docids, [3, 2, 1, 0])

    def testDeterministicLastTies(self):
        rf = DeterministicRankingFunction(-1, self.weights, ties="last")
        rf.init_ranking(self.query)
        self.assertEqual(rf.docids, [1, 3, 2, 0])

    def testProbabilisticDeterministic(self):
        rf = ProbabilisticRankingFunction(3, self.weights)
        rf.init_ranking(self.query)
        self.assertIn([rf.next_det() for _ in
            range(self.query.get_document_count())], [[1, 2, 3, 0],
                [1, 3, 2, 0]])

    def testProbabilisticSum(self):
        rf = ProbabilisticRankingFunction(3, self.weights)
        rf.init_ranking(self.query)
        self.assertAlmostEqual(1, sum([rf.get_document_probability(i) for i in
            range(self.query.get_document_count())]))

    def testProbabilisticDocs(self):
        rf = ProbabilisticRankingFunction(3, self.weights)
        rf.init_ranking(self.query)
        for _ in range(self.query.get_document_count()):
            docids = list(rf.docids)
            docid = rf.next()
            self.assertIn(docid, docids)
            self.assertNotIn(docid, rf.docids)

    def testProbabilisticProbabilities(self):
        rf = ProbabilisticRankingFunction(3, self.weights)
        rf.init_ranking(self.query)
        self.assertAlmostEqual(0.0132678, rf.get_document_probability(0))
        self.assertAlmostEqual(0.8491400, rf.get_document_probability(1))

if __name__ == '__main__':
        unittest.main()
