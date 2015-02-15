'''
Created on 12 jan. 2015

@author: Harrie, Jos
'''
import cStringIO
import random, argparse, os

from lerot.comparison.ProbabilisticInterleave import ProbabilisticInterleave
import lerot.comparison.ProbabilisticMultileave as ml
import lerot.comparison.SampleBasedProbabilisticMultileave as sbml
from lerot.comparison.TeamDraftMultileave import TeamDraftMultileave
import lerot.environment.CascadeUserModel as CascadeUserModel
import lerot.evaluation.NdcgEval as NdcgEval
import lerot.query as qu
import lerot.ranker.ProbabilisticRankingFunction as rnk
import numpy as np


description = "Script for experiments for probabilistic multileaving."

parser = argparse.ArgumentParser(description=description)
parser.add_argument("FOLD_PATH", help="path to folder containing train.txt, test.txt and vali.txt")
parser.add_argument("click_model", help="click model to use")
parser.add_argument("experiment_type", default="sensitivity", help="experiment type (bias or sensitivity) ")

args = parser.parse_args()

PATH_TEST_QUERIES  = os.path.join(args.FOLD_PATH,'test.txt')
PATH_VALI_QUERIES  = os.path.join(args.FOLD_PATH,'vali.txt')
PATH_TRAIN_QUERIES = os.path.join(args.FOLD_PATH,'train.txt')


class Experiment(object):

    # 64 features as in NP2003
    # k = ranking length
    def __init__(self, feature_sets, n_features=64, cutoff=10, click_model="navigational", experiment_type="sensitivity"):
        self.n_rankers  = len(feature_sets)
        self.n_features = n_features

        self.cutoff = cutoff
        self.train_queries = qu.load_queries(PATH_TRAIN_QUERIES, self.n_features)
        self.test_queries = qu.load_queries(PATH_TEST_QUERIES, self.n_features)

        self.samplemultil  = sbml.SampleBasedProbabilisticMultileave()

        self.multil        = ml.ProbabilisticMultileave()
        self.multil_nonbin = ml.ProbabilisticMultileave("-c True")
        self.interl = ProbabilisticInterleave('--aggregate binary')
        self.TeamDraftMultileave = TeamDraftMultileave()

        self.allrankers = [rnk("1", "random", self.n_features)
                        for _ in range(self.n_rankers)]

        for feature_ids,ranker in zip(feature_sets,self.allrankers):
            weights = np.zeros(self.n_features)
            for fid in feature_ids:
                weights[fid] = 1
            ranker.update_weights(weights)

        if experiment_type == "sensitivity":
            ndcg = NdcgEval()
        
            average_ndcgs = np.zeros((self.n_rankers))
            for query in self.test_queries:
                for i, ranker in enumerate(self.allrankers):
                    ranker.init_ranking(query)
                    average_ndcgs[i] += ndcg.get_value(ranker.get_ranking(),
                                                       query.get_labels().tolist(),
                                                       None, self.cutoff)
            average_ndcgs /= len(self.test_queries)

            self.all_true_pref = np.zeros((self.n_rankers, self.n_rankers))
            for i in range(self.n_rankers):
                for j in range(self.n_rankers):
                    self.all_true_pref[i, j] = 0.5 * (average_ndcgs[i] -
                                                  average_ndcgs[j]) + 0.5
        elif experiment_type == "bias":
            click_model = "random"
            self.all_true_pref = np.zeros((self.n_rankers, self.n_rankers))
            for i in range(self.n_rankers):
                for j in range(self.n_rankers):
                    self.all_true_pref[i, j] = 0.5

        if click_model=="navigational":
            click_str="--p_click 0:.05, 1:0.95 --p_stop  0:.2, 1:.5"
        elif click_model=="perfect":
            click_str="--p_click 0:.0, 1:1. --p_stop  0:.0, 1:.0"
        elif click_model=="informational":
            click_str="--p_click 0:.4, 1:.9 --p_stop  0:.1, 1:.5"
        elif click_model=="random":
            click_str="--p_click 0:.5, 1:.5 --p_stop  0:.5, 1:.5"

        self.user_model = CascadeUserModel(click_str)

    def run(self, n_impressions, n_rankers):
        self.n_rankers  = n_rankers
        self.rankerids = sorted(random.sample(range(len(self.allrankers)), self.n_rankers))
        self.rankers = [self.allrankers[i] for i in self.rankerids]
        
        self.true_pref = np.zeros((self.n_rankers, self.n_rankers))
        
        for inew, i in enumerate(self.rankerids):
            for jnew, j in enumerate(self.rankerids):
                self.true_pref[inew, jnew] = self.all_true_pref[i, j]

        error       = np.zeros(len(self.rankers))
        total_pm    = np.zeros((self.n_rankers, self.n_rankers))
        total_spm   = np.zeros((self.n_rankers, self.n_rankers))
        total_td    = np.zeros((self.n_rankers, self.n_rankers))
        total_pi    = np.zeros((self.n_rankers, self.n_rankers))
        count_pi    = np.zeros((self.n_rankers, self.n_rankers))
        # total_pm_nb = np.zeros((self.n_rankers, self.n_rankers))
        ave_nb_cred = np.zeros((self.n_rankers))
        
        for i in range(1,n_impressions+1):
            pm_preferences, td_preferences, ((pi_r1, pi_r2), pi_creds), sbpm_pref = self.impression()
            total_spm += sbpm_pref
            total_pm  += pm_preferences
            total_td  += td_preferences
            total_pi[pi_r1][pi_r2] +=  1-pi_creds
            total_pi[pi_r2][pi_r1] +=  pi_creds
            count_pi[pi_r1][pi_r2] += 1
            count_pi[pi_r2][pi_r1] += 1

            # ave_nb_cred += pm_nonbin_creds
            # total_pm_nb =  self.preferencesFromCredits((1-ave_nb_cred/i))

            # may be usefull for later debugging
            # print 
            # print pm_nonbin_creds
            # print ave_nb_cred/i
            # print
            # print self.preferencesFromCredits(ave_nb_cred/i)
            # print
            # print self.true_pref
            # print

            print i,

            for score in [ self.preference_error(matrix) for matrix in [total_pm/i,
                                total_td/i, total_pi/count_pi, 1-total_spm/i]]:
                print score,
            print

        total_spm   /= n_impressions
        total_pm    /= n_impressions
        total_td    /= n_impressions
        total_pi    /= count_pi

        return [ self.preference_error(matrix) for matrix in [total_pm, total_td, total_pi, 1-total_spm]]

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
        # pm_nonbin_creds = self.impression_probabilisticMultileave(query,False)

        pm_preferences = self.preferencesFromCredits(pm_creds)
        td_preferences = self.preferencesFromCredits(td_creds)

        return pm_preferences, td_preferences, ((pi_r1, pi_r2), pi_creds), self.impression_sampleProbabilisticMultileave(query)

    def impression_sampleProbabilisticMultileave(self, query):
        
        ranking, _ = self.samplemultil.multileave(self.rankers, query, self.cutoff)
        clicks     = self.user_model.get_clicks(ranking, query.get_labels())
        creds      = self.samplemultil.infer_outcome(ranking, self.rankers, clicks,
                                              query)
        return 1-creds


    def impression_probabilisticMultileave(self, query, binary=True):
        if binary:
            ranking, _ = self.multil.multileave(self.rankers, query, self.cutoff)
        else:
            ranking, _ = self.multil_nonbin.multileave(self.rankers, query, self.cutoff)
        clicks = self.user_model.get_clicks(ranking, query.get_labels())
        if binary:
            creds = self.multil.infer_outcome(ranking, self.rankers, clicks,
                                              query)
        else:
            creds = self.multil_nonbin.infer_outcome(ranking, self.rankers, clicks,
                                              query)
        return creds

    def impression_probabilisticInterleave(self, query):
        [r1, r2] = random.sample(self.rankers, 2)
        ranking, _ = self.interl.interleave(r1, r2, query, self.cutoff)
        clicks = self.user_model.get_clicks(ranking, query.get_labels())
        creds = self.interl.infer_outcome(ranking, (_, r1, r2),
                                          clicks, query)
        return creds/2.+0.5, (self.rankers.index(r1), self.rankers.index(r2))

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
        preferences = np.zeros((self.n_rankers,self.n_rankers))
        for i in range(self.n_rankers):
            for j in range(i+1):
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
    # for our output we don't want to see the zero divisions
    np.seterr("ignore");
    ranker_feature_sets = [
                             range(11,16), #TF-IDF
                             range(21,26), #BM25
                             range(36,41), #LMIR
                             [41,42],      #SiteMap
                             [49,50]       #HITS
                            ]
    experiment = Experiment(ranker_feature_sets, click_model=args.click_model, experiment_type=args.experiment_type)
    for i in range(25):
        print "RUN", i
        for name in ["probablistic_multi", "teamdraft_multi", "probabilistic_inter", "sample_probablistic_multi",]:
            print name,
        print
        experiment.run(1000, 5)
        print
