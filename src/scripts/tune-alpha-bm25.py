import sys, random
import include
import retrieval_system, environment, evaluation, query
import numpy as np

#Give alphas/deltas for b
alphas = [0.01, 0.05, 0.1, 0.5, 1.0]
deltas = [0.05, 0.1, 0.5, 1.0, 2.0]
reps = 5
nrqueries = 200

#alphas = [0.75, 1.0, 1.25, 1.5]
#deltas = [1.75, 2.0, 2.25]
#reps = 5
#nrqueries = 500


factor_k1 = 25
factor_k3 = .1

user_model = environment.CascadeUserModel('--p_click 0:0.0,1:1 --p_stop 0:0.0,1:0.0')
evaluator = evaluation.NdcgEval()

training_queries = query.load_queries(sys.argv[1], 64)
test_queries = query.load_queries(sys.argv[2], 64)


def run(alpha, delta):
    results = []
    for _ in range(reps):
        #learner = retrieval_system.ListwiseLearningSystem(64, '-w random -c comparison.ProbabilisticInterleave -r ranker.ProbabilisticRankingFunction -s 3 ranker.model.BM25 -d %.2f -a %.2f' % (delta, alpha))
        learner = retrieval_system.ListwiseLearningSystemWithCandidateSelection(64, '--num_repetitions 10 --num_candidates 6 --history_length 10 --select_candidate select_candidate_repeated -w random -c comparison.ProbabilisticInterleave --ranker ranker.ProbabilisticRankingFunction --ranker_args 3 ranker.model.BM25 -d %s -a %s' % (delta, alpha))
        for _ in range(nrqueries):
            q = training_queries[random.choice(training_queries.keys())]
            l = learner.get_ranked_list(q)
            c = user_model.get_clicks(l, q.get_labels())
            s = learner.update_solution(c)
        r = evaluator.evaluate_all(s, test_queries)
        results.append(r)
    return results


from multiprocessing import Pool
pool = Pool(processes=20)

results = {}
for alpha in alphas:
    a_k1 = alpha * factor_k1
    a_k3 = alpha * factor_k3
    a_b = alpha
    a = ",".join([str(x) for x in [a_k1, a_k3, a_b]])
    results[a] = {}
    for delta in deltas:
        d_k1 = delta * factor_k1
        d_k3 = delta * factor_k3
        d_b = delta
        d = ",".join([str(x) for x in [d_k1, d_k3, d_b]])
        results[a][d] = pool.apply_async(run, (a, d))

for a in sorted(results.keys()):
    for d in sorted(results[a].keys()):
        r = results[a][d].get()
        print a, d, np.mean(r), np.std(r)

pool.close()
pool.join()

