'''
Created on 12 jan. 2015

@author: Jos
'''
import cStringIO
import unittest

import lerot.comparison.ProbabilisticMultileave as ml
import lerot.query as qu
import lerot.ranker.ProbabilisticRankingFunction as rnk


PATH_QUERIES = '../../../data/Fold1/train.txt'


class Test(unittest.TestCase):

    def setUp(self):
        self.test_num_features = 6
        self.test_query = _readQueries(PATH_QUERIES)

    def testListCreation(self, n_rankers=3):
        multil = ml.ProbabilisticMultileave()

        query_fh = cStringIO.StringIO(self.test_query)
        queries = qu.Queries(query_fh, self.test_num_features)

        query = queries[queries.keys()[0]]
        query_fh.close()

        ranker_arg_str = ['ranker.model.BM25', '1']
            # second arg corresponds to ranker_type..
        ties = "random"
        feature_count = None
        rankers = [rnk(ranker_arg_str, ties, feature_count)
                   for _ in range(n_rankers)]
        length = 10
        (docs, _) = multil.multileave(rankers, query, length)

        foundDocs = [d.docid for d in docs]
        existingDocs = [q.docid for q in query.get_docids()]
        assert(set(foundDocs).issubset(set(existingDocs)))
        assert(len(foundDocs) == length)
        assert(len(foundDocs) == len(set(foundDocs)))  # No duplicates

    def testInfer_Outcome(self):
        pass


def _readQueries(path, numberOfLines=100):
    with open(path, "r") as myfile:
        data = myfile.read()
    data = '\n'.join(data.splitlines()[0:numberOfLines])
    return data


if __name__ == "__main__":
    unittest.main()
