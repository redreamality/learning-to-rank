'''
Created on 12 jan. 2015

@author: Jos
'''
import cStringIO
import unittest

import lerot.comparison.ProbabilisticMultileave as ml
import lerot.ranker.ProbabilisticRankingFunction as rnk
import lerot.query as qu


class Test(unittest.TestCase):

    def setUp(self):
        self.test_num_features = 6
        self.test_query = """
        4 qid:1 1:2.6 2:1 3:2.1 4:0 5:2 6:1.4 # highly relevant
        1 qid:1 1:1.2 2:1 3:2.9 4:0 5:2 6:1.9 # bad
        0 qid:1 1:0.5 2:1 3:2.3 4:0 5:2 6:5.6 # not relevant
        0 qid:1 1:0.5 2:1 3:2.3 4:0 5:2 6:5.6 # not relevant
        """

    def testListCreation(self, n_rankers=3):
        multil = ml.ProbabilisticMultileave()

        query_fh = cStringIO.StringIO(self.test_query)
        # TODO: load queries of database
        queries = qu.Queries(query_fh, self.test_num_features)
        query = queries['1']
        query_fh.close()

        ranker_arg_str = ['ranker.model.BM25', '1']
            # second arg corresponds to ranker_type..
        ties = "random"
        feature_count = None 
        rankers = [rnk(ranker_arg_str, ties, feature_count)
                   for _ in range(n_rankers)]
        length = 10
        (docs, (a, rankers)) = multil.multileave(rankers, query, length)
        assert(set([d.docid for d in docs]) == set(range(4)))

    def testInfer_Outcome(self):
        pass


if __name__ == "__main__":
    unittest.main()
