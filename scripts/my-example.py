import sys, random, os, time
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


test_queries = lerot.query.load_queries(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))+'\\data\\MQ2007\\Fold2\\test.txt', 46)
evaluation = lerot.evaluation.NdcgEval()
#print training_queries.__dict__
learner = lerot.retrieval_system.ListwiseLearningSystem(training_queries.__num_features__, '-w random -c comparison.TeamDraft -r ranker.DeterministicRankingFunction -s 3 -d 0.1 -a 0.01')
user_model = lerot.environment.LivingLabsRealUser(KEY, training_queries.__doc_ids__)#CascadeUserModel('--p_click 0:0.0,1:.5,2:1.0 --p_stop 0:0.0,1:0.0,2:0.0')
runid = 0
qlen = len(training_queries.keys())
while True:
    uploads = {}
    runid += 1
    counter = 0
    for qid in training_queries.keys()[:]:
        counter += 1
        print qid, counter, 'of', qlen
        #['__feature_vectors__', '__docids__', '__labels__', '__qid__', '__comments__']
        q = training_queries[qid]
        l = learner.get_ranked_list(q)
        payload, the_time = user_model.upload_run(q, l, runid)
        uploads[qid] = {'ranked_lerot_list':l, 'payload':payload, 'upload_time':the_time}
        time.sleep(0.1)

    shuffledQids = training_queries.keys()[:]
    random.shuffle(shuffledQids)
    while True:
        counter = 0
        time.sleep(0.1)
        for qid in shuffledQids:
            counter += 1
            print qid, counter, 'of', qlen
            q = training_queries[qid]
            c = user_model.get_clicks(uploads[qid]['ranked_lerot_list'], q.get_labels(), 
                                      query=q, ranker_list=uploads[qid]['payload'], upload_time=uploads[qid]['upload_time'])
            if c != None: break
        if c != None:
            print c
            break     
    s = learner.update_solution(c)
    eval = evaluation.evaluate_all(s, test_queries)
    print eval
    with open('evaluation', 'a') as f:
        f.write(str(eval)+'\n')
