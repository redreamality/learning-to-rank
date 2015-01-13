'''
Created on 12 jan. 2015

@author: Jos
'''
import cStringIO
import random
import unittest

import lerot.comparison.ProbabilisticMultileave as ml
import lerot.query as qu
import lerot.ranker.ProbabilisticRankingFunction as rnk
import lerot.environment.CascadeUserModel as CascadeUserModel
import numpy as np


PATH_QUERIES = 'data/Fold1/train.txt'


class Temp_doc():
    def __init__(self,id):
        self.id = id
    def get_id(self):
        return self.id


class Test(unittest.TestCase):

    def setUp(self):
        self.test_num_features = 6
        self.test_queries = _readQueries(PATH_QUERIES)

    def step1_ListCreation(self, n_rankers=3):
        print('Testing step 1: creation of multileaved list')
        multil = ml.ProbabilisticMultileave()

        query_fh = cStringIO.StringIO(self.test_queries)
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
        (createdList, _) = multil.multileave(rankers, query, length)

        foundDocs = [d.docid for d in createdList]
        existingDocs = [q.docid for q in query.get_docids()]
        assert(set(foundDocs).issubset(set(existingDocs)))
        assert(len(foundDocs) == length)
        assert(len(foundDocs) == len(set(foundDocs)))  # No duplicates

        # For next step:
        self.foundDocs = createdList
        self.rankers = rankers
        self.query = query
        self.multil = multil

    def step2_InferOutcome(self):
        print('Testing step 2: infer_outcome')
        l = self.foundDocs
        rankers = self.rankers
        clicks = [random.randint(0, 1) for _ in range(len(l))]
        user_model = CascadeUserModel("--p_click 0:.0, 1:1.0"
                              " --p_stop 0:.0, 1:.0");
       
        query = self.query

        #TODO wtf does CascadeUserModel want as input?!? not a list of docids
        c = user_model.get_clicks([Temp_doc(docid) for docid in l],query.get_labels())

        creds = self.multil.infer_outcome(l, rankers, c, query)

        print "Multileaved list:", l
        print "Clicks on list:  ", c
        print "Credit:          ", creds

        assert(len(creds) == len(l))
        assert(np.allclose(sum(creds), 1.))
        # TODO: unittest if values make sense

    def testSteps(self):
        steps = [self.step1_ListCreation, self.step2_InferOutcome]
        for step in steps:
            step()


def _readQueries(path, numberOfLines=100):
    with open(path, "r") as myfile:
        data = myfile.read()
    data = '\n'.join(data.splitlines()[0:numberOfLines])
    return data


if __name__ == "__main__":
    unittest.main()
