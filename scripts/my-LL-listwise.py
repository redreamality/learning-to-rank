import sys, random, os, time, copy
import numpy as np
import include
import lerot.retrieval_system, lerot.environment, lerot.evaluation, lerot.query
try:
   import cPickle as pickle
except:
   import pickle



for fold in range(5):
    fold = fold+1
    for repetition in range(2):
        KEY = "3061C7A142E10020-SNHXGT52ORZRQ3SC"
        if os.path.isfile('queriesLLPickle.c'):
            with open('queriesLLPickle.c', 'rb') as fp:
                training_queries = pickle.load(fp)
        else:
            print 'Retrieving Queries Data'
            training_queries = lerot.query.load_livinglabs_queries(KEY)
            with open('queriesLLPickle.c', 'wb') as fp:
                pickle.dump(training_queries, fp)
        
        
        #test_queries = lerot.query.load_queries(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))+'\\data\\MQ2007\\Fold2\\test.txt', 46)
        #evaluation = lerot.evaluation.NdcgEval()
        #print training_queries.__dict__
        learner = lerot.retrieval_system.ListwiseLearningSystem(training_queries.__num_features__, '-w random -c comparison.TeamDraft -r ranker.DeterministicRankingFunction -s 3 -d 0.1 -a 0.01')
        user_model = lerot.environment.LivingLabsRealUser(KEY, training_queries.__doc_ids__)#CascadeUserModel('--p_click 0:0.0,1:.5,2:1.0 --p_stop 0:0.0,1:0.0,2:0.0')
        runid = 0
        while True:
            uploads = {}
            runid += 1
            print 'iteration: ', runid
            firstTime = True
            counter = 0
            print 'Uploading rankings'
            start = time.time()
        
            for qid in training_queries.keys()[:]:
                counter += 1
                print qid, counter, 'of', len(training_queries.keys())
                #['__feature_vectors__', '__docids__', '__labels__', '__qid__', '__comments__']
                q = training_queries[qid]
                l = learner.get_ranked_list(q, firstTime)
                payload, the_time = user_model.upload_run(q, l, runid)
                firstTime = False
                uploads[qid] = {'current_l':l,
                                'payload':payload,
                                'upload_time':the_time,
                                'current_context': learner.current_context,
                                'current_query': q
                                }
                if counter>=10 and counter % 10 == 0:
                    end = time.time()
                    print end - start, 'seconds'
                    #sys.exit()
                time.sleep(0.01)
        
            shuffledQids = training_queries.keys()[:]
            random.shuffle(shuffledQids)
            print 'Checking for click feedback'
            while True:
                time.sleep(0.01)
                for qid in shuffledQids:
                    q = training_queries[qid]
                    c = user_model.get_clicks(uploads[qid]['current_l'], q.get_labels(), 
                                              query=q, ranker_list=uploads[qid]['payload'], upload_time=uploads[qid]['upload_time'])
                    if isinstance(c, np.ndarray):break
                if isinstance(c, np.ndarray):
                    print c
                    break   
            s = learner.update_solution(c)
            learner.get_ranked_list(q, firstTime)
            learner.current_l = uploads[qid]['current_l']
            learner.current_context = uploads[qid]['current_context']
            
            
            eval = evaluation.evaluate_all(s, test_queries)
            out_path = 'LL-listwise_evaluation_data\\fold'+str(fold)+'_'+(str(repetition))
            evalString += str(eval)
            evalString += '\n'
            if runid % 100 == 0:#runid >= 100 and 
                print time.time() - start
                with open(out_path, 'a') as f:
                    f.write(str(evalString))
                    evalString = ''
                with open('LLlistwiselearnerPickle.c', 'wb') as fp:
                    pickle.dump(learner, fp)
            if runid == 1000:
                break
