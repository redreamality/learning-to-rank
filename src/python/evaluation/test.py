import unittest
import sys
import os
import cStringIO
import numpy as np

sys.path.insert(0, os.path.abspath('..'))

import query as qu
from LetorNdcgEval import LetorNdcgEval
from NdcgEval import NdcgEval


class TestEvaluation(unittest.TestCase):

    def setUp(self):
        self.test_num_features = 6
        test_query = """
        4 qid:1 1:2.6 2:1 3:2.1 4:0 5:2 6:1.4 # highly relevant
        1 qid:1 1:1.2 2:1 3:2.9 4:0 5:2 6:1.9 # bad
        0 qid:1 1:0.5 2:1 3:2.3 4:0 5:2 6:5.6 # not relevant
        0 qid:1 1:0.5 2:1 3:2.3 4:0 5:2 6:5.6 # not relevant
        """

        self.query_fh = cStringIO.StringIO(test_query)
        self.queries = qu.Queries(self.query_fh, self.test_num_features)
        self.query = self.queries['1']

        zero_weight_str = "0 0 0 0 0 0"
        self.zero_weights = np.asarray([float(x) for x in
            zero_weight_str.split()])

        weight_str = "0 0 1 0 0 0"
        self.weights = np.asarray([float(x) for x in weight_str.split()])

    def testGetDcg(self):
        ev = NdcgEval()
        self.assertAlmostEqual(ev.get_dcg([0, 0, 1, 0]), 0.5)
        self.assertAlmostEqual(ev.get_dcg([0, 1, 0, 0]), 0.6309298)
        self.assertAlmostEqual(ev.get_dcg([0, 0, 0, 0]), 0)

    def testGetNdcg(self):
        ev = NdcgEval()
        self.assertAlmostEqual(ev.get_ndcg([0, 0, 1, 0], [1, 0, 0, 0]), 0.5)
        self.assertAlmostEqual(ev.get_ndcg([0, 1, 0, 0], [1, 0, 0, 0]),
            0.6309298)
        self.assertAlmostEqual(ev.get_ndcg([0, 0, 0, 0], [1, 0, 0, 0]), 0)

    def testGetLetorDcg(self):
        ev = LetorNdcgEval()
        self.assertAlmostEqual(ev.get_dcg([0, 0, 1, 0]), 0.6309298)
        self.assertEqual(ev.get_dcg([0, 1, 0, 0]), 1)
        self.assertEqual(ev.get_dcg([0, 0, 0, 0]), 0)

    def testGetLetorNdcg(self):
        ev = LetorNdcgEval()
        self.assertAlmostEqual(ev.get_ndcg([0, 0, 1, 0], [1, 0, 0, 0]),
            0.6309298)
        self.assertEqual(ev.get_ndcg([0, 1, 0, 0], [1, 0, 0, 0]), 1)
        self.assertEqual(ev.get_ndcg([0, 0, 0, 0], [1, 0, 0, 0]), 0)

    def testEvaluateRankingFirstZero(self):
        ev = LetorNdcgEval()
        dot_prod = np.dot(self.query.get_feature_vectors(),
            self.zero_weights.transpose())
        ranking = ev._sort_docids_by_score(self.query.get_docids(),
            dot_prod, ties="first")
        self.assertEquals(1, ev.evaluate_ranking(ranking, self.query))

    def testEvaluateRankingLastZeroWithCutoff(self):
        ev = NdcgEval()
        dot_prod = np.dot(self.query.get_feature_vectors(),
            self.zero_weights.transpose())
        ranking = ev._sort_docids_by_score(self.query.get_docids(),
            dot_prod, ties="last")
        self.assertAlmostEquals(0.4452805, ev.evaluate_ranking(ranking,
            self.query, cutoff=10))

    def testEvaluateRankingLastZeroWithCutoffLetor(self):
        ev = LetorNdcgEval()
        dot_prod = np.dot(self.query.get_feature_vectors(),
            self.zero_weights.transpose())
        ranking = ev._sort_docids_by_score(self.query.get_docids(),
            dot_prod, ties="last")
        self.assertAlmostEquals(0.5081831, ev.evaluate_ranking(ranking,
            self.query, cutoff=10))

    def testEvaluateRankingRandomZero(self):
        test_ndcgs = {"1.0000": 0, "0.9769": 0, "0.9688": 0, "0.6540": 0,
            "0.6227": 0, "0.5312": 0, "0.5082": 0}
        trials = 0
        MAX_TRIALS = 1000
        ev = LetorNdcgEval()
        dot_prod = np.dot(self.query.get_feature_vectors(),
            self.zero_weights.transpose())
        while trials < MAX_TRIALS and 0 in test_ndcgs.values():
            trials += 1
            ranking = ev._sort_docids_by_score(self.query.get_docids(),
                dot_prod, ties="random")
            observed_ndcg = "%.4f" % ev.evaluate_ranking(ranking, self.query)
            if observed_ndcg in test_ndcgs:
                test_ndcgs[observed_ndcg] += 1
            else:
                print "unknown ndcg: ", observed_ndcg
        print "Observed all test ndcgs within %d trials." % trials
        for ndcg, count in test_ndcgs.items():
            self.assertNotEqual(0, count, "Test failed for %s" % ndcg)

    def testEvaluateRankingFirst(self):
        ev = LetorNdcgEval()
        dot_prod = np.dot(self.query.get_feature_vectors(),
            self.weights.transpose())
        ranking = ev._sort_docids_by_score(self.query.get_docids(),
            dot_prod, ties="first")
        self.assertEquals(0.53125, ev.evaluate_ranking(ranking, self.query),
            msg="ranking " + ",".join(str(d) for d in ranking))

    def testEvaluateRankingLast(self):
        ev = LetorNdcgEval()
        dot_prod = np.dot(self.query.get_feature_vectors(),
            self.weights.transpose())
        ranking = ev._sort_docids_by_score(self.query.get_docids(),
            dot_prod, ties="last")
        self.assertAlmostEquals(0.53125, ev.evaluate_ranking(ranking,
            self.query), msg="ranking " + ",".join(str(d) for d in ranking))

    def testEvaluateOneFirstZero(self):
        ev = LetorNdcgEval()
        self.assertEquals(1, ev.evaluate_one(self.zero_weights, self.query,
            ties="first"))

    def testEvaluateOneLastZero(self):
        ev = LetorNdcgEval()
        self.assertAlmostEquals(0.5081831, ev.evaluate_one(self.zero_weights,
            self.query, ties="last"))


if __name__ == '__main__':
    unittest.main()
