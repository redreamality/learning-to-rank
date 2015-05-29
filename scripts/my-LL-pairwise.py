import sys, random, os, time, copy
import numpy as np
import include
import lerot.retrieval_system, lerot.environment, lerot.evaluation, lerot.query
try:
   import cPickle as pickle
except:
   import pickle


KEY = "3061C7A142E10020-SNHXGT52ORZRQ3SC"
if os.path.isfile('queriesLLPickle.c'):
    with open('queriesLLPickle.c', 'rb') as fp:
        training_queries = pickle.load(fp)
else:
    print 'Retrieving Queries Data'
    training_queries = lerot.query.load_livinglabs_queries(KEY)
    with open('queriesLLPickle.c', 'wb') as fp:
        pickle.dump(training_queries, fp)



directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))+'/output_data'
if not os.path.exists(directory):
    os.makedirs(directory)
directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))+'/output_data/pairwise_LL_evaluation_data'
if not os.path.exists(directory):
    os.makedirs(directory)


existingData = False
if os.path.isfile(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))+'/output_data/pairwise_LL_evaluation_data/data'):
    data_out = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))+'/output_data/pairwise_LL_evaluation_data//data'
    existingData = True
if existingData == True:
    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))+'/output_data/pairwise_LL_evaluation_data/evaluation.c', 'r') as fp:
        print 'Lodaded existing evaluation'
	evaluation = pickle.load(fp)
    with open(data_out, 'r') as f:
        thelist = f.read().split('\n')
        index = len(thelist)
	print 'Lodaded existing index value from list'
	print index
    thelearner = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))+'/output_data/pairwise_LL_evaluation_data//learnerPickle.c'
    with open(thelearner, 'rb') as fp:
        learner = pickle.load(fp)
	print 'loaded learner'
else:
    print 'Did not find existing Data'
    learner = lerot.retrieval_system.PairwiseLearningSystem(training_queries.__num_features__, "--init_weights random --epsilon 0.0 --eta 0.001 --ranker ranker.DeterministicRankingFunction --ranker_tie first")        
    evaluation = lerot.evaluation.LivingLabsEval()
    index = 1
user_model = lerot.environment.LivingLabsRealUser(KEY, training_queries.__doc_ids__)#CascadeUserModel('--p_click 0:0.0,1:.5,2:1.0 --p_stop 0:0.0,1:0.0,2:0.0')




evalString = ''
for repetition in range(index, 1001):
    uploads = {}
    print 'iteration: ', repetition
    print 'Uploading rankings....'
    theid = 'pairwise'+str(repetition)
    for qid in training_queries.keys()[:]:
        q = training_queries[qid]
        l = learner.get_ranked_list(q)
        payload, the_time = user_model.upload_run(q, l, theid)
        uploads[qid] = {'current_l':l,
                        'payload':payload,
                        'upload_time':the_time,
                        'current_query': q,
                        'runid':theid
                        }


    shuffledQids = training_queries.keys()[:]
    random.shuffle(shuffledQids)
    print 'Checking for click feedback'
    while True:
        for qid in shuffledQids:   
            q = training_queries[qid]
            feedback, c = user_model.get_clicks(uploads[qid]['current_l'], q.get_labels(), 
                                                    query=q, ranker_list=uploads[qid]['payload'], upload_time=uploads[qid]['upload_time'], run_id=uploads[qid]['runid'])
            if isinstance(c, np.ndarray):break
        if isinstance(c, np.ndarray):
            print c
            break   
    s = learner.update_solution(c)
    learner.get_ranked_list(q)
    learner.current_l = uploads[qid]['current_l']
        
        
    win = user_model.get_win(q, feedback, uploads[qid]['payload'])
    evaluation.update_score(win)
    eval = str(evaluation.get_win()).strip('[]')
    print eval
    print evaluation.get_performance()
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))+'/output_data/pairwise_LL_evaluation_data/data'
    evalString += eval
    evalString += '\n'
    if repetition >= 10 and repetition % 10 == 0:#
        with open(out_path, 'a') as f:
            f.write(str(evalString))
            evalString = ''
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))+'/output_data/pairwise_LL_evaluation_data/'+'learnerPickle.c', 'wb') as fp:
            pickle.dump(learner, fp)
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))+'/output_data/pairwise_LL_evaluation_data/evaluation.c', 'wb') as fp:
            pickle.dump(evaluation, fp)