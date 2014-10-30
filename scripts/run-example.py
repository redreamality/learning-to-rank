import sys, random
import include
import lerot.retrieval_system, lerot.environment, lerot.evaluation, lerot.query


learner = lerot.retrieval_system.ListwiseLearningSystem(46, '-w random -c comparison.TeamDraft -r ranker.DeterministicRankingFunction -s 3 -d 0.1 -a 0.01')
user_model = lerot.environment.CascadeUserModel('--p_click 0:0.0,1:.5,2:1.0 --p_stop 0:0.0,1:0.0,2:0.0')
evaluation = lerot.evaluation.NdcgEval()
training_queries = lerot.query.load_queries(sys.argv[1], 46)
test_queries = lerot.query.load_queries(sys.argv[2], 46)
while True:
    q = training_queries[random.choice(training_queries.keys())]
    l = learner.get_ranked_list(q)
    c = user_model.get_clicks(l, q.get_labels())
    s = learner.update_solution(c)
    print evaluation.evaluate_all(s, test_queries)
