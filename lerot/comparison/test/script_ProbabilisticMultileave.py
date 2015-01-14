'''
Created on 12 jan. 2015

@author: Harrie, Jos
'''
import cStringIO
import random

from lerot.comparison.ProbabilisticInterleave import ProbabilisticInterleave
import lerot.comparison.ProbabilisticMultileave as ml
from lerot.comparison.TeamDraftMultileave import TeamDraftMultileave
import lerot.environment.CascadeUserModel as CascadeUserModel
import lerot.evaluation.NdcgEval as NdcgEval
import lerot.query as qu
import lerot.ranker.ProbabilisticRankingFunction as rnk
import numpy as np


PATH_TEST_QUERIES  = 'data/Fold1/test.txt'
PATH_VALI_QUERIES  = 'data/Fold1/vali.txt'
PATH_TRAIN_QUERIES = 'data/Fold1/train.txt'


class Experiment(object):

    # 64 features as in NP2003
    # k = ranking length
    def __init__(self, feature_sets, n_features=64, cutoff=10, click_model="navigational"):
        self.n_rankers  = len(n_rankers)
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

        self.multil        = ml.ProbabilisticMultileave()
        self.multil_nonbin = ml.ProbabilisticMultileave("-c True")
        self.interl = ProbabilisticInterleave('--aggregate binary')
        self.TeamDraftMultileave = TeamDraftMultileave()

        self.rankers = [rnk("1", "random", self.n_features)
                        for _ in range(self.n_rankers)]

        for feature_ids,ranker in zip(feature_sets,self.rankers):
            weights = np.zeros(self.n_features)
            for fid in feature_ids:
                weights[fid] = 1
            ranker.update_weights(weights)

        ndcg = NdcgEval()
        average_ndcgs = np.zeros((self.n_rankers))
        for query in self.test_queries:
            for i, ranker in enumerate(self.rankers):
                ranker.init_ranking(query)
                average_ndcgs[i] += ndcg.get_value(ranker.get_ranking(),
                                                   query.get_labels().tolist(),
                                                   None, self.cutoff)
        average_ndcgs /= len(self.test_queries)

        self.true_pref = np.zeros((self.n_rankers, self.n_rankers))
        for i in range(self.n_rankers):
            for j in range(self.n_rankers):
                self.true_pref[i, j] = 0.5 * (average_ndcgs[i] -
                                              average_ndcgs[j]) + 0.5

        if click_model=="navigational":
            click_str="--p_click 0:.05, 1:0.95 --p_stop  0:.2, 1:.5"
        elif click_model=="perfect":
            click_str="--p_click 0:.0, 1:1. --p_stop  0:.0, 1:.0"
        # navigational click model
        self.user_model = CascadeUserModel(click_str)

    def run(self, n_impressions):
        error       = np.zeros(len(self.rankers))
        total_pm    = np.zeros((self.n_rankers, self.n_rankers))
        total_td    = np.zeros((self.n_rankers, self.n_rankers))
        total_pi    = np.zeros((self.n_rankers, self.n_rankers))
        count_pi    = np.zeros((self.n_rankers, self.n_rankers))
        total_pm_nb = np.zeros((self.n_rankers, self.n_rankers))
        
        for i in range(n_impressions):
            pm_preferences, td_preferences, ((pi_r1, pi_r2), pi_creds), pm_nonbin_creds = self.impression()
            total_pm += pm_preferences
            total_td += td_preferences
            total_pm_nb += pm_nonbin_creds
            total_pi[pi_r1][pi_r2] =  pi_creds
            total_pi[pi_r2][pi_r1] = -pi_creds
            count_pi[pi_r1][pi_r2] += 1
            count_pi[pi_r2][pi_r1] += 1

            print [ self.preference_error(matrix) for matrix in [total_pm/i,
                                total_td/i, total_pm_nb/i, total_pi/count_pi]]

        total_pm    /= n_impressions
        total_td    /= n_impressions
        total_pm_nb /= n_impressions
        total_pi    /= count_pi

        return [ self.preference_error(matrix) for matrix in [total_pm, total_td, total_pm_nb, total_pi]]

    def impression(self):
        '''
        TODO: docstring

        RETURN:
        - probabilistic multileaving: preference matrix
        - team draft: preference matrix
        - probabilisticInterleaving: tuple of (tuple with index of rankers),
          credits)
        '''
        query = self.train_queries[random.choice(self.train_queries.keys())]

        pm_creds = self.impression_probabilisticMultileave(query)
        pi_creds, (pi_r1, pi_r2) = \
            self.impression_probabilisticInterleave(query)
        td_creds = self.impression_teamDraftMultileave(query)
        pm_nonbin_creds = self.impression_probabilisticMultileave(query,False)

        pm_preferences = self.preferencesFromCredits(pm_creds)
        td_preferences = self.preferencesFromCredits(td_creds)

        return pm_preferences, td_preferences, ((pi_r1, pi_r2), pi_creds), pm_nonbin_creds

    def impression_probabilisticMultileave(self, query, binary=True):
        if binary:
            ranking, _ = self.multil.multileave(self.rankers, query, self.cutoff)
        else:
            ranking, _ = self.multil_nonbin.multileave(self.rankers, query, self.cutoff)
        clicks = self.user_model.get_clicks(ranking, query.get_labels())
        creds = self.multil.infer_outcome(ranking, self.rankers, clicks,
                                          query)
        return creds

    def impression_probabilisticInterleave(self, query):
        [r1, r2] = random.sample(self.rankers, 2)
        ranking, _ = self.interl.interleave(r1, r2, query, self.cutoff)
        clicks = self.user_model.get_clicks(ranking, query.get_labels())
        creds = self.interl.infer_outcome(ranking, (_, r1, r2),
                                          clicks, query)
        return creds, (self.rankers.index(r1), self.rankers.index(r2))

    def impression_teamDraftMultileave(self, query):
        ranking, a = self.TeamDraftMultileave.interleave(self.rankers, query,
                                                         self.cutoff)
        clicks = self.user_model.get_clicks(ranking, query.get_labels())
        creds = self.TeamDraftMultileave.infer_outcome(ranking, a, clicks,
                                                          query)
        return creds

    def preference_error(self, preference_matrix):
        error = 0
        for i in range(self.n_rankers):
            for j in range(self.n_rankers):
                if j != i and np.sign(preference_matrix[i, j] - 0.5) != \
                        np.sign(self.true_pref[i, j] - 0.5):
                    error += 1.
        return error / (self.n_rankers * (self.n_rankers - 1))

    def preferencesFromCredits(self, creds):
        '''
        ARGS:
        - creds: a list with credits for each of the rankers

        RETURNS:
        - preferences: list of list containing 1 if ranker_row is better than
          ranker_collumn, 0 if he is worse, 0.5 if he is equal
        '''
        n = len(self.rankers)
        preferences = [[None] * n] * n
        for i in range(len(self.rankers)):
            for j in range(len(self.rankers)):
                if creds[i] > creds[j]:
                    pref = 1
                elif creds[i] < creds[j]:
                    pref = 0
                else:
                    pref = 0.5
                preferences[i][j] = pref
                preferences[j][i] = 1 - pref
        return preferences


def _readQueries(path):
    with open(path, "r") as myfile:
        data = myfile.read()
    return data


if __name__ == "__main__":
    ranker_feature_sets = [
                             range(11,16), #TF-IDF
                             range(21,26), #BM25
                             range(36,41), #LMIR
                             [41,42],      #SiteMap
                             [49,50]       #HITS
                            ]
    experiment = Experiment(ranker_feature_sets)
    for i in range(10):
        print "RUN", i
        print experiment.run(500)
