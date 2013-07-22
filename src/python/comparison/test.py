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

import query as qu
from ranker import DeterministicRankingFunction, ProbabilisticRankingFunction

from BalancedInterleave import BalancedInterleave
from TeamDraft import TeamDraft
from DocumentConstraints import DocumentConstraints
from ProbabilisticInterleave import ProbabilisticInterleave

from HistBalancedInterleave import HistBalancedInterleave
from HistTeamDraft import HistTeamDraft
from HistDocumentConstraints import HistDocumentConstraints
from HistProbabilisticInterleave import HistProbabilisticInterleave

from ExploitativeProbabilisticInterleave import \
    ExploitativeProbabilisticInterleave


class TestEvaluation(unittest.TestCase):

    def setUp(self):
        self.test_num_features = 6
        test_query = """
        1 qid:1 1:2.6 2:1 3:2.1 4:0 5:2 6:1.4 # relevant
        1 qid:1 1:1.2 2:1 3:2.9 4:0 5:2 6:1.9 # relevant
        0 qid:1 1:0.5 2:1 3:2.3 4:0 5:2 6:5.6 # not relevant
        0 qid:1 1:0.5 2:1 3:2.3 4:0 5:2.1 6:5.6 # not relevant
        """

        self.query_fh = cStringIO.StringIO(test_query)
        self.queries = qu.Queries(self.query_fh, self.test_num_features)
        self.query = self.queries['1']

        zero_weight_str = "0 0 0 0 0 0"
        self.zero_weights = np.asarray([float(x) for x in
            zero_weight_str.split()])

        # results in ranking: 1, 3, 2, 0
        weight_str_1 = "0 0 1 0 1 0"
        self.weights_1 = np.asarray([float(x) for x in weight_str_1.split()])
        weight_str_2 = "1 0 0 0 1 0"
        self.weights_2 = np.asarray([float(x) for x in weight_str_2.split()])

    def testBalancedInterleave(self):
        bi = BalancedInterleave()
        r1 = DeterministicRankingFunction(None, self.weights_1)
        r2 = DeterministicRankingFunction(None, self.weights_2)
        (interleaved_list, assignments) = bi.interleave(r1, r2, self.query, 10)
        self.assertIn(interleaved_list.tolist(), [[0, 1, 2, 3], [1, 0, 2, 3],
            [0, 1, 3, 2], [1, 0, 3, 2]])
        self.assertEqual(assignments[0].tolist(), [1, 3, 2, 0])
        self.assertEqual(assignments[1].tolist(), [0, 1, 3, 2])
        o = bi.infer_outcome([1, 0, 3, 2], [[1, 3, 2, 0], [0, 1, 3, 2]],
            [1, 0, 0, 0], self.query)
        self.assertEqual(o, -1, "l1 should win (1), o = %g" % o)
        o = bi.infer_outcome([1, 0, 3, 2], [[1, 3, 2, 0], [0, 1, 3, 2]],
            [1, 0, 1, 0], self.query)
        self.assertEqual(o, -1, "l1 should win (2), o = %g" % o)
        o = bi.infer_outcome([1, 0, 3, 2], [[1, 2, 3, 0], [0, 1, 3, 2]],
            [1, 0, 1, 0], self.query)
        self.assertEqual(o, 0, "The rankers should tie (1), o = %g" % o)
        o = bi.infer_outcome([0, 1, 2, 3], [[0, 1, 2, 3], [1, 2, 3, 0]],
            [0, 1, 0, 1], self.query)
        self.assertEqual(o, 1, "l1 should win, o = %g" % o)
        o = bi.infer_outcome([1, 0, 2, 3], [[0, 1, 2, 3], [1, 2, 3, 0]],
            [0, 1, 0, 1], self.query)
        self.assertEqual(o, 0, "The rankers should tie (2), o = %g" % o)
        o = bi.infer_outcome([0, 2, 1, 3], [[3, 0, 1, 2], [1, 3, 2, 0]],
            [1, 0, 1, 0], self.query)
        self.assertEqual(o, -1, "l1 should win (3), o = %g" % o)
        o = bi.infer_outcome([0, 2, 1, 3], [[3, 0, 1, 2], [4, 3, 2, 0]],
            [1, 0, 1, 0], self.query)
        self.assertEqual(o, -1, "l1 should win (4), o = %g" % o)

    def testHistBalancedInterleave(self):
        hbi = HistBalancedInterleave()
        r1 = DeterministicRankingFunction(None, self.weights_1)
        r1_test = DeterministicRankingFunction(None, self.weights_1)
        r2 = DeterministicRankingFunction(None, self.weights_2)
        self.assertEqual(hbi._get_assignment(r1, r1_test, self.query,
            4)[0].tolist(), [1, 3, 2, 0])
        self.assertEqual(hbi._get_assignment(r1, r1_test, self.query,
            4)[1].tolist(), [1, 3, 2, 0])  # r1
        self.assertEqual(hbi._get_assignment(r1, r2, self.query,
            4)[1].tolist(), [0, 1, 3, 2])  # r2
        o = hbi.infer_outcome([1, 0, 3, 2], ["src a is ignored"], [1, 0, 0, 0],
            r1, r2, self.query)
        self.assertEqual(o, -1, "Same as original, l1 should win, o = %g" % o)
        o = hbi.infer_outcome([1, 0, 3, 2], ["src a is ignored"], [1, 0, 0, 0],
            r2, r1, self.query)
        self.assertEqual(o, 1, "Different from original, l2 should win, "
            "o = %g" % o)
        o = hbi.infer_outcome([1, 0, 3, 2], ["src a is ignored"], [1, 0, 0, 0],
            r1_test, r1, self.query)
        self.assertEqual(o, 0, "Same ranking - tie (1), o = %g" % o)
        o = hbi.infer_outcome([2, 0, 3, 1], ["src a is ignored"], [1, 1, 0, 0],
            r1, r2, self.query)
        self.assertEqual(o, 0, "Same ranking - tie (2), o = %g" % o)
        o = hbi.infer_outcome([2, 0, 3, 4], ["src a is ignored"], [1, 1, 0, 0],
            r1, r2, self.query)
        self.assertEqual(o, 0, "Same ranking - tie (3), o = %g" % o)

    def testDocumentConstraints(self):
        dc = DocumentConstraints()
        r1 = DeterministicRankingFunction(None, self.weights_1)
        r2 = DeterministicRankingFunction(None, self.weights_2)
        (interleaved_list, assignments) = dc.interleave(r1, r2, self.query, 10)
        self.assertIn(interleaved_list.tolist(), [[0, 1, 2, 3], [1, 0, 2, 3],
            [0, 1, 3, 2], [1, 0, 3, 2]])
        self.assertIn(assignments[0].tolist(), [[1, 2, 3, 0], [1, 3, 2, 0]])
        self.assertIn(assignments[1].tolist(), [[0, 1, 2, 3], [0, 1, 3, 2]])
        o = dc.infer_outcome([1, 0, 3, 2], [[1, 3, 2, 0], [0, 1, 3, 2]],
            [1, 0, 0, 0], self.query)
        self.assertEqual(o, -1, "l1 should win (1), o = %g" % o)
        o = dc.infer_outcome([1, 0, 3, 2], [[1, 3, 2, 0], [0, 1, 3, 2]],
            [0, 0, 0, 1], self.query)
        self.assertEqual(o, -1, "l1 should win (2), o = %g" % o)
        o = dc.infer_outcome([1, 0, 3, 2], [[1, 3, 2, 0], [0, 1, 3, 2]],
            [0, 1, 0, 0], self.query)
        self.assertEqual(o, 1, "l2 should win (1), o = %g" % o)
        o = dc.infer_outcome([1, 0, 3, 2], [[1, 0, 2, 3], [0, 1, 3, 2]],
            [0, 1, 0, 0], self.query)
        # constraints: 0 > 1, 0 > 3
        self.assertEqual(o, 1, "l2 should win (2), o = %g" % o)
        o = dc.infer_outcome([1, 0, 3, 2], [[1, 2, 0, 3], [1, 0, 2, 3]],
            [0, 1, 1, 0], self.query)
        # constraints: 0 > 1, 3 > 1, 0 > 2, 3 > 2
        self.assertEqual(o, 1, "l2 should win (3), o = %g" % o)
        o = dc.infer_outcome([1, 0, 3, 2], [[1, 3, 2, 0], [0, 1, 3, 2]],
            [0, 0, 0, 0], self.query)
        self.assertEqual(o, 0, "No winner when there are no clicks o = %g" % o)
        o = dc.infer_outcome([1, 0, 3, 2], [[1, 3, 2, 0], [0, 1, 3, 2]],
            [1, 1, 1, 1], self.query)
        self.assertEqual(o, 0, "No winner when all are clicked o = %g" % o)
        dc = DocumentConstraints("--constraints 1")
        o = dc.infer_outcome([1, 0, 3, 2], [[1, 0, 2, 3], [3, 0, 1, 2]],
            [0, 1, 0, 0], self.query)
        # constraint: 0 > 1
        self.assertEqual(o, 1, "l2 should win with one constraint, o = %g" % o)
        dc = DocumentConstraints("--constraints 2")
        o = dc.infer_outcome([1, 0, 3, 2], [[1, 0, 2, 3], [3, 0, 1, 2]],
            [0, 1, 0, 0], self.query)
        self.assertEqual(o, 0, "Tie with two constraint types (1), o = %g" % o)
        o = dc.infer_outcome([1, 0, 3, 2], [[1, 0, 2, 3], [1, 2, 0, 3]],
            [0, 1, 1, 0], self.query)
        # constraints: 0 > 1, 3 > 1, 3 > 2
        self.assertEqual(o, 0, "Tie with two constraint types (2), o = %g" % o)
        o = dc.infer_outcome([1, 0, 3, 2], [[1, 0, 4, 3], [1, 0, 3, 2]],
            [0, 1, 1, 0], self.query)
        self.assertEqual(o, 0, "Tie with two constraint types (3), o = %g" % o)
        o = dc.infer_outcome([1, 0, 3, 2], [[1, 0, 4, 3], [1, 0, 2, 3]],
            [0, 1, 1, 0], self.query)
        self.assertEqual(o, -1, "l1 should win with two constr., o = %g" % o)

    def testHistDocumentConstraints(self):
        hdc = HistDocumentConstraints()
        r1 = DeterministicRankingFunction(None, self.weights_1)
        r2 = DeterministicRankingFunction(None, self.weights_2)
        # results in assignments l1 = [1, 2, 3, 0] or [1, 3, 2, 0]
        # and l2 = [0, 1, 2, 3] or [0, 1, 3, 2]
        o = hdc.infer_outcome([1, 0, 3, 2], None, [1, 0, 0, 0], r1, r2,
            self.query)
        self.assertEqual(o, -1, "l1 should win, o = %g" % o)
        o = hdc.infer_outcome([2, 1, 3, 0], None, [1, 0, 0, 0], r1, r2,
            self.query)
        self.assertEqual(o, 0, "No winner, both have 1 > 2 (1), o = %g" % o)
        o = hdc.infer_outcome([2, 1, 4, 0], None, [1, 0, 0, 0], r1, r2,
            self.query)
        self.assertEqual(o, 0, "No winner, both have 1 > 2 (2), o = %g" % o)
        o = hdc.infer_outcome([2, 1, 3, 0], None, [0, 0, 0, 0], r1, r2,
            self.query)
        self.assertEqual(o, 0, "No winner when none are clicked, o = %g" % o)
        o = hdc.infer_outcome([2, 1, 3, 0], None, [1, 1, 1, 1], r1, r2,
            self.query)
        self.assertEqual(o, 0, "No winner when all are clicked, o = %g" % o)

    def testTeamDraftInterleave(self):
        td = TeamDraft(None)
        r1 = DeterministicRankingFunction(None, self.weights_1)
        r2 = DeterministicRankingFunction(None, self.weights_2)
        (interleaved_list, assignments) = td.interleave(r1, r2, self.query, 10)
        self.assertIn(interleaved_list.tolist(), [[0, 1, 2, 3], [1, 0, 2, 3],
            [0, 1, 3, 2], [1, 0, 3, 2]])
        self.assertIn(assignments.tolist(), [[0, 1, 0, 1], [1, 0, 1, 0],
            [1, 0, 0, 1], [0, 1, 1, 0]])

    def testHistTeamDraft_getPossibleAssignment(self):
        r1 = DeterministicRankingFunction(None, self.weights_1)
        r2 = DeterministicRankingFunction(None, self.weights_2)
        htd = HistTeamDraft(None)
        l = [0, 1, 3, 2]
        self.assertIn(htd._get_possible_assignment(l, r1, r2, self.query),
            [[1, 0, 0, 1], [1, 0, 1, 0]])
        l = [1, 0, 3, 2]
        self.assertIn(htd._get_possible_assignment(l, r1, r2, self.query),
            [[0, 1, 0, 1], [0, 1, 1, 0]])
        l = [1, 0, 2, 3]
        self.assertEquals(htd._get_possible_assignment(l, r1, r2, self.query),
            None)

    def testHistTeamDraft_getPossibleAssignment_randomization(self):
        r1 = DeterministicRankingFunction(None, self.weights_1)
        r2 = DeterministicRankingFunction(None, self.weights_2)
        htd = HistTeamDraft(None)
        l = [0, 1, 3, 2]
        test_assignments = {"1,0,0,1": 0, "1,0,1,0": 0}
        trials = 0
        MAX_TRIALS = 1000
        while trials < MAX_TRIALS and 0 in test_assignments.values():
            trials += 1
            observed_assignment = ",".join(str(a) for a in
                htd._get_possible_assignment(l, r1, r2, self.query))
            self.assertIn(observed_assignment, test_assignments.keys())
            test_assignments[observed_assignment] += 1
        for assignment, count in test_assignments.items():
            self.assertNotEqual(0, count, "Test failed for: %s" % assignment)

    def testHistTeamDraft(self):
        r1 = DeterministicRankingFunction(None, self.weights_1)
        r2 = DeterministicRankingFunction(None, self.weights_2)
        interleaved_list = [0, 1, 3, 2]
        htd = HistTeamDraft()
        self.assertEqual(htd.infer_outcome(interleaved_list, None,
            [0, 0, 0, 0], r1, r2, self.query), 0, "No clicks.")
        self.assertEqual(htd.infer_outcome(interleaved_list, None,
            [1, 0, 0, 0], r1, r2, self.query), 1, "Target rankers"
            " are the same as the original rankers, so ranker 2 has to win.")
        self.assertEqual(htd.infer_outcome(interleaved_list, None,
            [1, 0, 0, 0], r2, r1, self.query), -1, "Target rankers"
            " are switched, so ranker 1 has to win.")

    def testProbabilisticInterleaveWithDeterministicRankers(self):
        pi = ProbabilisticInterleave(None)
        # test a few possible interleavings
        r1 = DeterministicRankingFunction(None, self.weights_1)
        r2 = DeterministicRankingFunction(None, self.weights_2)
        test_lists = {"0,1,3,2": 0, "1,0,3,2": 0, "1,3,0,2": 0, "1,3,2,0": 0}
        trials = 0
        MAX_TRIALS = 10000
        while trials < MAX_TRIALS and 0 in test_lists.values():
            trials += 1
            (l, a) = pi.interleave(r1, r2, self.query, 10)
            list_str = ",".join(str(a) for a in l.tolist())
            self.assertIn(list_str, test_lists.keys())
            test_lists[list_str] += 1
        for list_str, count in test_lists.items():
            self.assertNotEqual(0, count,
                "Interleave failed for: %s" % list_str)
        # test interleaving outcomes
        context = (None, r1, r2)
        self.assertEqual(pi.infer_outcome([0, 1, 2, 3], context, [0, 0, 0, 0],
            self.query), 0, "No clicks, outcome should be 0.")
        self.assertEqual(pi.infer_outcome([0, 1, 2, 3], context, [1, 0, 0, 0],
            self.query), 0, "No possible assignment, outcome should be 0.")
        o = pi.infer_outcome([1, 0, 3, 2], context, [1, 0, 0, 0], self.query)
        self.assertAlmostEquals(o, -0.0625, 4,
            "Ranker 1 should win (o = %.4f)." % o)
        o = pi.infer_outcome([0, 1, 3, 2], context, [1, 0, 0, 0], self.query)
        self.assertAlmostEquals(o, 0.0625, 4,
            "Ranker 2 should win (o = %.4f)." % o)
        # test get_probability_of_list
        p = pi.get_probability_of_list([1, 0, 3, 2], context, self.query)
        self.assertEqual(p, 0.25, "Probability of the most "
            "likely list. p = %g" % p)

    def testProbabilisticInterleave(self):
        pi = ProbabilisticInterleave(None)
        r1 = ProbabilisticRankingFunction(3, self.weights_1)
        r2 = ProbabilisticRankingFunction(3, self.weights_2)
        context = (None, r1, r2)
        # test get_probability_of_list
        p = pi.get_probability_of_list([1, 0, 3, 2], context, self.query)
        self.assertAlmostEquals(p, 0.182775, 6, "Probability of the most "
            "likely list. p = %.6f" % p)
        # test a few possible interleavings
        test_lists = {"0,1,2,3": 0, "0,1,3,2": 0, "0,2,1,3": 0, "0,2,3,1": 0,
                      "0,3,1,2": 0, "0,3,2,1": 0, "1,0,2,3": 0, "1,0,3,2": 0,
                      "1,2,0,3": 0, "1,2,3,0": 0, "1,3,0,2": 0, "1,3,2,0": 0,
                      "2,0,1,3": 0, "2,0,3,1": 0, "2,1,0,3": 0, "2,1,3,0": 0,
                      "2,3,0,1": 0, "2,3,1,0": 0, "3,0,1,2": 0, "3,0,2,1": 0,
                      "3,1,0,2": 0, "3,1,2,0": 0, "3,2,0,1": 0, "3,2,1,0": 0}
        trials = 0
        MAX_TRIALS = 100000
        while trials < MAX_TRIALS and 0 in test_lists.values():
            trials += 1
            (l, _) = pi.interleave(r1, r2, self.query, 10)
            list_str = ",".join(str(a) for a in l.tolist())
            self.assertIn(list_str, test_lists.keys())
            test_lists[list_str] += 1
        for list_str, count in test_lists.items():
            self.assertNotEqual(0, count,
                "Interleave failed for: %s" % list_str)
        # test interleaving outcomes
        self.assertEqual(pi.infer_outcome([0, 1, 2, 3], context, [0, 0, 0, 0],
            self.query), 0, "No clicks, outcome should be 0.")
        o = pi.infer_outcome([1, 0, 3, 2], context, [1, 0, 0, 0], self.query)
        self.assertAlmostEquals(o, -0.0486, 4,
            "Ranker 1 should win (o = %.4f)." % o)
        o = pi.infer_outcome([0, 1, 3, 2], context, [1, 0, 0, 0], self.query)
        self.assertAlmostEquals(o, 0.0606, 4,
            "Ranker 2 should win (o = %.4f)." % o)
        # from the example in CIKM 2011
        weight_str_1 = "0 0 1 0 -1 0"
        weights_1 = np.asarray([float(x) for x in weight_str_1.split()])
        weight_str_2 = "1 0 0 0 -1 0"
        weights_2 = np.asarray([float(x) for x in weight_str_2.split()])
        r1 = ProbabilisticRankingFunction(3, weights_1)
        r2 = ProbabilisticRankingFunction(3, weights_2)
        context = (None, r2, r1)
        o = pi.infer_outcome([0, 1, 2, 3], context, [0, 1, 1, 0], self.query)
        self.assertAlmostEquals(o, 0.0046, 4,
            "Ranker 2 should win again (o = %.4f)." % o)
        # click on one before last document
        o = pi.infer_outcome([3, 1, 0, 2], context, [0, 0, 1, 0], self.query)
        self.assertAlmostEquals(o, -0.0496, 4,
            "Ranker 1 should win with click on doc 0 (o = %.4f)." % o)
        # click on last document
        o = pi.infer_outcome([3, 1, 2, 0], context, [0, 0, 0, 1], self.query)
        self.assertAlmostEquals(o, 0.0, 4,
            "Tie for click on last doc (o = %.4f)." % o)

    def testHistProbabilisticInterleave(self):
        r1 = ProbabilisticRankingFunction(3, self.weights_1)
        r2 = ProbabilisticRankingFunction(3, self.weights_2)
        hpi = HistProbabilisticInterleave(None)
        a = ([0, 1, 1, 0], r1, r2)
        self.assertEqual(hpi.infer_outcome([0, 1, 2, 3], a, [0, 0, 0, 0],
            r1, r2, self.query), 0, "No clicks, outcome should be 0.")
        o = hpi.infer_outcome([1, 0, 3, 2], a, [1, 0, 0, 0], r1, r2,
            self.query)
        self.assertAlmostEquals(o, -0.0486, 4, "Same target as original "
            "rankers. Ranker 1 should win (o = %.4f)." % o)
        o = hpi.infer_outcome([1, 0, 3, 2], a, [1, 0, 0, 0], r2, r1,
            self.query)
        self.assertAlmostEquals(o, 0.0486, 4, "Target rankers switched. "
            "Ranker 2 should win (o = %.4f)." % o)
        test_r1 = ProbabilisticRankingFunction(3, self.weights_1)
        a = ([0, 1, 1, 0], r1, test_r1)
        o = hpi.infer_outcome([1, 0, 3, 2], a, [1, 0, 0, 0], r2, r1,
            self.query)
        self.assertAlmostEquals(o, 0.1542, 4, "Same original ranker. "
            "Ranker 2 should win (o = %.4f)." % o)

    def testHistProbabilisticInterleaveWithoutMarginalization(self):
        r1 = ProbabilisticRankingFunction(3, self.weights_1)
        r2 = ProbabilisticRankingFunction(3, self.weights_2)
        hpiIs = HistProbabilisticInterleave("--biased False "
            "--marginalize False")
        # test get_probability_of_list_and_assignment
        p = hpiIs._get_probability_of_list_and_assignment([1, 3, 2, 0],
            [0, 0, 0, 0], r1, r2, self.query)
        self.assertAlmostEqual(p, 0.026261, 6, "Most likely list for ranker 1."
            " p = %e" % p)
        # test overall outcomes
        a = ([0, 1, 1, 0], r1, r2)
        self.assertEqual(hpiIs.infer_outcome([0, 1, 2, 3], a, [0, 0, 0, 0],
            r1, r2, self.query), 0, "No clicks, outcome should be 0.")
        o = hpiIs.infer_outcome([1, 0, 3, 2], a, [1, 0, 0, 0], r1, r2,
            self.query)
        self.assertEquals(o, -1, "Same original and target pair. "
            "Ranker 1 should win (o = %d)." % o)
        test_r1 = ProbabilisticRankingFunction(3, self.weights_1)
        a = ([0, 1, 1, 0], r1, test_r1)
        o = hpiIs.infer_outcome([1, 0, 3, 2], a, [1, 0, 0, 0], r2, r1,
            self.query)
        self.assertAlmostEquals(o, -0.1250, 4, "Different original pair. "
            "Ranker 1 should win (o = %.4f)." % o)

    def testExploitativeProbabilisticInterleave(self):
        r1 = ProbabilisticRankingFunction(1, self.weights_1)
        r2 = ProbabilisticRankingFunction(1, self.weights_2)
        r1.init_ranking(self.query)
        r2.init_ranking(self.query)
        epi = ExploitativeProbabilisticInterleave("--exploration_rate=0.5")
        (docids, probs) = epi._get_document_distribution(r1, r2)
        exp_docids = [1, 0, 3, 2]
        exp_probs = [0.36, 0.3, 0.2, 0.14]
        self._prob_doc_test_helper(docids, exp_docids, probs, exp_probs)

    def testExploitativeProbabilisticInterleaveThreeDocs(self):
        epi = ExploitativeProbabilisticInterleave("--exploration_rate=0.5")
        # prepare rankers
        r1 = ProbabilisticRankingFunction(1, self.weights_1)
        r2 = ProbabilisticRankingFunction(1, self.weights_2)
        r1.init_ranking(self.query)
        r2.init_ranking(self.query)
        r1.rm_document(0)
        r2.rm_document(0)
        # test after document 0 was removed
        (docids, probs) = epi._get_document_distribution(r1, r2)
        exp_docids = [1, 3, 2]
        exp_probs = [0.5034965, 0.29020979, 0.20629371]
        self._prob_doc_test_helper(docids, exp_docids, probs, exp_probs)
        # prepare rankers
        r1.init_ranking(self.query)
        r2.init_ranking(self.query)
        r1.rm_document(3)
        r2.rm_document(3)
        # test after document 3 was removed
        (docids, probs) = epi._get_document_distribution(r1, r2)
        exp_docids = [1, 0, 2]
        exp_probs = [0.45864662, 0.36466165, 0.17669173]
        self._prob_doc_test_helper(docids, exp_docids, probs, exp_probs)

    def testExploitativeProbabilisticInterleaveTwoDocs(self):
        # prepare rankers
        r1 = ProbabilisticRankingFunction(1, self.weights_1)
        r2 = ProbabilisticRankingFunction(1, self.weights_2)
        r1.init_ranking(self.query)
        r2.init_ranking(self.query)
        r1.rm_document(1)
        r2.rm_document(1)
        r1.rm_document(3)
        r2.rm_document(3)
        # test after 1 and 3 were removed
        epi = ExploitativeProbabilisticInterleave("--exploration_rate=0.5")
        (docids, probs) = epi._get_document_distribution(r1, r2)
        exp_docids = [0, 2]
        exp_probs = [0.61428571, 0.38571429]
        self._prob_doc_test_helper(docids, exp_docids, probs, exp_probs)

    def testExploitativeProbabilisticInterleaveExploit(self):
        r1 = ProbabilisticRankingFunction(1, self.weights_1)
        r2 = ProbabilisticRankingFunction(1, self.weights_2)
        # exploration rate = 0.1
        epi = ExploitativeProbabilisticInterleave("--exploration_rate=0.1")
        r1.init_ranking(self.query)
        r2.init_ranking(self.query)
        (docids, probs) = epi._get_document_distribution(r1, r2)
        exp_docids = [1, 3, 2, 0]
        exp_probs = [0.456, 0.232, 0.156, 0.156]
        self._prob_doc_test_helper(docids, exp_docids, probs, exp_probs)
        # exploration rate = 0.0
        epi = ExploitativeProbabilisticInterleave("--exploration_rate=0.0")
        r1.init_ranking(self.query)
        r2.init_ranking(self.query)
        (docids, probs) = epi._get_document_distribution(r1, r2)
        exp_docids = [1, 3, 2, 0]
        exp_probs = [0.48, 0.24, 0.16, 0.12]
        self._prob_doc_test_helper(docids, exp_docids, probs, exp_probs)

    def _prob_doc_test_helper(self, docids, exp_docids, probs, exp_probs):
        for r, (docid, prob) in enumerate(zip(docids, probs)):
            self.assertEquals(docid, exp_docids[r], "Docid %d did not match "
                "expected %d at rank %d" % (docid, exp_docids[r], r))
            self.assertAlmostEquals(prob, exp_probs[r], 6, "Prob %g did not "
                "match expected %g at rank %d" % (prob, exp_probs[r], r))

    def testExploitativeProbabilisticInterleaveInterleave(self):
        r1 = ProbabilisticRankingFunction(1, self.weights_1)
        r2 = ProbabilisticRankingFunction(1, self.weights_2)
        epi = ExploitativeProbabilisticInterleave("--exploration_rate=0.5")
        r1.init_ranking(self.query)
        r2.init_ranking(self.query)
        (l, (r1_ret, r2_ret)) = epi.interleave(r1, r2, self.query, 4)
        self.assertEqual(r1, r1_ret, "r1 is just passed through.")
        self.assertEqual(r2, r2_ret, "r2 is just passed through.")
        self.assertEqual(len(l), 4, "interleave produces a list of length 4.")
        self.assertTrue(0 in l, "document 0 is in l.")
        self.assertTrue(1 in l, "document 0 is in l.")
        self.assertTrue(2 in l, "document 0 is in l.")
        self.assertTrue(3 in l, "document 0 is in l.")

        observed_l = {}
        for _ in range(0, 100):
            (l, (r1_ret, r2_ret)) = epi.interleave(r1, r2, self.query, 4)
            l_str = " ".join([str(docid) for docid in l])
            if not l_str in observed_l:
                observed_l[l_str] = 1
            else:
                observed_l[l_str] += 1
        self.assertIn("0 1 2 3", observed_l, "List was observed: 0 1 2 3.")
        self.assertIn("1 0 3 2", observed_l, "List was observed: 0 1 2 3.")
        self.assertIn("3 1 2 0", observed_l, "List was observed: 0 1 2 3.")
        self.assertIn("2 1 0 3", observed_l, "List was observed: 0 1 2 3.")

    def testGetSourceProbabilityOfList(self):
        r1 = ProbabilisticRankingFunction(1, self.weights_1)
        r2 = ProbabilisticRankingFunction(1, self.weights_2)
        # with exploration rate 0.5
        epi = ExploitativeProbabilisticInterleave("--exploration_rate=0.5")
        p = epi._get_source_probability_of_list([1, 0, 3, 2], (None, r1, r2),
            self.query)
        self.assertAlmostEquals(0.090916137, p, 8, "Obtained p = %.g" % p)
        # with exploration rate 0.1
        epi = ExploitativeProbabilisticInterleave("--exploration_rate=0.1")
        p = epi._get_source_probability_of_list([1, 0, 3, 2], (None, r1, r2),
            self.query)
        self.assertAlmostEquals(0.073751736, p, 8, "Obtained p = %.g" % p)

    def testInferOutcomeBiased(self):
        r1 = ProbabilisticRankingFunction(1, self.weights_1)
        r2 = ProbabilisticRankingFunction(1, self.weights_2)
        epi = ExploitativeProbabilisticInterleave("--exploration_rate=0.1 "
            "--biased=True")
        outcome = epi.infer_outcome([1, 0, 3, 2], (None, r1, r2), [0, 1, 0, 0],
            self.query)
        self.assertAlmostEquals(0.029049296, outcome, 8,
            "Obtained outcome = %.8f" % outcome)

    def testInferOutcomeUnbiased(self):
        r1 = ProbabilisticRankingFunction(1, self.weights_1)
        r2 = ProbabilisticRankingFunction(1, self.weights_2)
        epi = ExploitativeProbabilisticInterleave("--exploration_rate=0.1")
        outcome = epi.infer_outcome([1, 0, 3, 2], (None, r1, r2), [0, 1, 0, 0],
            self.query)
        self.assertAlmostEquals(0.03581, outcome, 8,
            "Obtained outcome = %.8f" % outcome)

if __name__ == '__main__':
    unittest.main()
