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

sys.path.insert(0, os.path.abspath('..'))

from lerot import query
from numpy import array
from ListwiseLearningSystem import ListwiseLearningSystem
from PairwiseLearningSystem import PairwiseLearningSystem


class TestListwiseLearning(unittest.TestCase):
    def setUp(self):
        # initialize query
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
        # initialize listwise learner
        self.learner = ListwiseLearningSystem(self.test_num_features,
            "--init_weights 0,0,1,0,0,0 --delta 1.0 --alpha 0.01 --ranker "
            "ranker.ProbabilisticRankingFunction --ranker_args 3 --ranker_tie "
            "first --comparison comparison.ProbabilisticInterleaveWithHistory"
            " --comparison_args \"--history_length 10 --biased true\"")

    def testRanker(self):
        self.learner.get_ranked_list(self.query)


class TestPairwiseLearning(unittest.TestCase):
    def setUp(self):
        # initialize query
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
        # initialize pairwise learner
        self.learner = PairwiseLearningSystem(self.test_num_features,
            "--init_weights 0,0,1,0,0,0 --epsilon 0.0 --eta 0.001 --ranker "
            "ranker.DeterministicRankingFunction --ranker_tie first")

    def testGetSolution(self):
        self.assertEqual(list([0, 0, 1, 0, 0, 0]),
            list(self.learner.get_solution()))

    def testGetRankedList(self):
        self.assertEqual(list([1, 2, 3, 0]),
            list(self.learner.get_ranked_list(self.query)))

    def testUpdateSolution(self):
        self.learner.get_ranked_list(self.query)
        new_weights = self.learner.update_solution(array([0, 1, 0, 0]))
        # check values one by one - needed due to floating point prec. diffs
        self.assertEqual(6, len(new_weights))
        for x, y in zip([-0.0007, 0, 0.9994, 0, 0, 0.0037], new_weights):
            self.assertEqual(round(x, 4), round(y, 4),
                "mismatch between %.4f - %.4f" % (x, y))

if __name__ == '__main__':
        unittest.main()
