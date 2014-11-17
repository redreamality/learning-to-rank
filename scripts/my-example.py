import sys, random, os
import include
import lerot.retrieval_system, lerot.environment, lerot.evaluation, lerot.query
try:
   import cPickle as pickle
except:
   import pickle


KEY = "3061C7A142E10020-SNHXGT52ORZRQ3SC"
if os.path.isfile('queriesPickle.c'):
    with open('queriesPickle.c', 'rb') as fp:
        training_queries = pickle.load(fp)
else:
    print 'Retrieving Queries Data'
    training_queries = lerot.query.load_livinglabs_queries(KEY)
    with open('queriesPickle.c', 'wb') as fp:
        pickle.dump(training_queries, fp)
#print training_queries.__dict__
learner = lerot.retrieval_system.ListwiseLearningSystem(training_queries.__num_features__, '-w random -c comparison.TeamDraft -r ranker.DeterministicRankingFunction -s 3 -d 0.1 -a 0.01')
user_model = lerot.environment.LivingLabsRealUser(KEY, training_queries.__doc_ids__)#CascadeUserModel('--p_click 0:0.0,1:.5,2:1.0 --p_stop 0:0.0,1:0.0,2:0.0')
runid = 0
qids = 'Ls-q142'

while True:
    runid += 1

    print runid, qids
    #['__feature_vectors__', '__docids__', '__labels__', '__qid__', '__comments__']
    q = training_queries[qids]
    l = learner.get_ranked_list(q)
    print l
    payload = user_model.upload_run(q, l, runid)
    c = user_model.get_clicks(l, q.get_labels(), query=q, ranker_list=payload)#upload_time=the_time
    print c
    s = learner.update_solution(c)
        
        
    '''for qids in training_queries.keys():
        print runid, qids
        #['__feature_vectors__', '__docids__', '__labels__', '__qid__', '__comments__']
        q = training_queries[qids]
        l = learner.get_ranked_list(q)
        payload = user_model.upload_run(q, l, runid)
        c = user_model.get_clicks(l, q.get_labels(), query=q, ranker_list=payload)#upload_time=the_time
        print c
        s = learner.update_solution(c)'''