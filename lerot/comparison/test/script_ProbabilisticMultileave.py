'''
Created on 12 jan. 2015

@author: Harrie, Jos
'''
import cStringIO
import random

import lerot.comparison.ProbabilisticMultileave as ml
import lerot.query as qu
import lerot.ranker.ProbabilisticRankingFunction as rnk
import lerot.environment.CascadeUserModel as CascadeUserModel
import lerot.evaluation.NdcgEval as NdcgEval
import numpy as np


PATH_TEST_QUERIES = 'data/Fold1/test.txt'
PATH_VALI_QUERIES = 'data/Fold1/vali.txt'
PATH_TRAIN_QUERIES = 'data/Fold1/train.txt'


class Experiment(object):

    # 64 features as in NP2003
    # k = ranking length
    def __init__(self, n_rankers, n_features=64, cutoff=10, click_model="navigational"):
        self.n_rankers = n_rankers
        self.n_features = n_features
        self.cutoff = cutoff
        train_raw = _readQueries(PATH_TRAIN_QUERIES) + '\n' \
                                        + _readQueries(PATH_VALI_QUERIES)
        test_raw = _readQueries(PATH_TRAIN_QUERIES)

        query_fh = cStringIO.StringIO(train_raw)
        self.train_queries = qu.Queries(query_fh, self.n_features)
        query_fh.close()

        query_fh = cStringIO.StringIO(test_raw)
        self.test_queries = qu.Queries(query_fh, self.n_features)
        query_fh.close()

        arg_str = ""
        # if (credits):
        #     arg_str = "-c True"
        self.multil = ml.ProbabilisticMultileave(arg_str)

        self.rankers = [rnk("1", "random", self.n_features)
                        for _ in range(self.n_rankers)]

        weights = np.zeros(self.n_features)
        # weights[np.random.randint(self.n_features)] = 1
        # weights[np.random.randint(self.n_features)] = 1
        # weights[np.random.randint(self.n_features)] = 1
        # weights[np.random.randint(self.n_features)] = 1
        # weights[np.random.randint(self.n_features)] = 1

        for ranker in self.rankers:
            weights = np.zeros(self.n_features)
            for i in range(5):
                weights[np.random.randint(self.n_features)] = 1
            # weights[40] = 1
            ranker.update_weights(weights)
        # random.shuffle(self.rankers)

        ndcg = NdcgEval()
        average_ndcgs = np.zeros((self.n_rankers))
        for query in self.test_queries:
            for i, ranker in enumerate(self.rankers):
                ranker.init_ranking(query)
                average_ndcgs[i] += ndcg.get_value(ranker.get_ranking(), query.get_labels().tolist(),None,self.cutoff)
        average_ndcgs /= len(self.test_queries)

        self.true_pref = np.zeros((self.n_rankers,self.n_rankers))
        for i in range(self.n_rankers):
            for j in range(self.n_rankers):
                self.true_pref[i,j] = 0.5*(average_ndcgs[i] - average_ndcgs[j]) + 0.5

        click_str=
        if click_model=="navigational":
            click_str="--p_click 0:.05, 1:0.95 --p_stop  0:.2, 1:.5"
        elif click_model=="perfect":
            click_str="--p_click 0:.0, 1:1. --p_stop  0:.0, 1:.0"
        # navigational click model
        self.user_model = CascadeUserModel(click_str)

    def run(self):
        total_creds = np.zeros(len(self.rankers))
        count = 0

        for _ in range(500):
            creds = self.impression()
            total_creds += np.array(creds)
            if sum(creds) > 0:
                count += 1

        return total_creds / count

    def impression(self):
        # select query
        query = self.train_queries[random.choice(self.train_queries.keys())]

        # probabilistic multileave
        (ranking, _) = self.multil.multileave(self.rankers, query, self.cutoff)
        clicks = self.user_model.get_clicks(ranking, query.get_labels())
        pm_creds = self.multil.infer_outcome(ranking, self.rankers, clicks, query)

        #probabilistic interleave
        # pick pair of rankers
        # interleave
        # observe clicks
        # assign credits

        #team draft multileave
        # multileave
        # observe clicks
        # assign credits

        # merge all credits into preference matrix


        return pm_creds

    def preference_error(self, preference_matrix):
        error = 0
        for i in range(self.n_rankers):
            for j in range(self.n_rankers):
                if np.sign(preference_matrix[i,j] - 0.5) != np.sign(preference_matrix[i,j] -0.5):
                    error += 1.
        return error/(self.n_rankers*(self.n_rankers-1))


def _readQueries(path):
    with open(path, "r") as myfile:
        data = myfile.read()
    data = '\n'.join(data.splitlines())
    return data


if __name__ == "__main__":
    experiment = Experiment(6)
    for i in range(10):
        print experiment.run()
