import sys, random, os, time, copy
import include
import numpy as np
import lerot.retrieval_system, lerot.environment, lerot.evaluation, lerot.query
try:
   import cPickle as pickle
except:
   import pickle


directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))+'\\output_data'
if not os.path.exists(directory):
    os.makedirs(directory)
directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))+'\\output_data\\listwise_local_evaluation_data'
if not os.path.exists(directory):
    os.makedirs(directory)

for fold in range(5):
    fold = fold+1
    for repetition in range(5):
        training_queries = lerot.query.load_queries(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))+'\\data\\MQ2007\\Fold'+str(fold)+'\\train.txt', 46)
        test_queries = lerot.query.load_queries(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))+'\\data\\MQ2007\\Fold'+str(fold)+'\\test.txt', 46)

        
        evaluation = lerot.evaluation.NdcgEval()
        learner = lerot.retrieval_system.ListwiseLearningSystem(46, '-w random -c comparison.TeamDraft -r ranker.DeterministicRankingFunction -s 3 -d 0.1 -a 0.01')
        user_model = lerot.environment.CascadeUserModel('--p_click 0:0.05,1:.5,2:0.95 --p_stop 0:0.2,1:0.5,2:0.9')
        runid = 0
        evalString = ''
        while True:
            runid += 1
            print 'Iteration:', runid, 'Repetition:', repetition, 'Fold:', fold
            uploads = {}
            firstTime = True
            for qid in training_queries.keys()[:]:
                q = training_queries[qid]
                l = learner.get_ranked_list(q, firstTime)
                firstTime = False
                uploads[qid] = {'current_l': l,
                                'current_context': learner.current_context,
                                'current_query': q
                                }
                
            shuffledQids = training_queries.keys()[:]
            random.shuffle(shuffledQids)#shuffle query ids list
            while True:
                for qid in shuffledQids:
                    q = training_queries[qid]
                    c = user_model.get_clicks(uploads[qid]['current_l'], q.get_labels())
                    if isinstance(c, np.ndarray):break
                if isinstance(c, np.ndarray):
                    break
        
            q = training_queries[qid]
            learner.get_ranked_list(q, firstTime)
            learner.current_l = uploads[qid]['current_l']
            learner.current_context = uploads[qid]['current_context']
            s = learner.update_solution(c)#update learner using clicks
            eval = evaluation.evaluate_all(s, test_queries)
            out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))+'\\output_data\\listwise_local_evaluation_data\\fold'+str(fold)+'_'+(str(repetition))
            evalString += str(eval)
            evalString += '\n'
            if runid >= 100 and runid % 100 == 0:
                with open(out_path, 'a') as f:
                    f.write(str(evalString))
                    evalString = ''
                with open(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))+'\\output_data\\listwise_local_evaluation_data\\'+'learnerPickle.c', 'wb') as fp:
                    pickle.dump(learner, fp)
            if runid == 1000:
                break
