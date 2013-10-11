import sys
import include
import evaluation, query, ranker
import numpy as np
from ranker.AbstractRankingFunction import AbstractRankingFunction

evaluation = evaluation.NdcgEval()
bm25ranker = AbstractRankingFunction(["ranker.model.BM25"], 'first', 3, sample="utils.sample_fixed")
queries = query.load_queries(sys.argv[1], 64)
#print evaluation.evaluate_all(bm25ranker, queries)

fh = open(sys.argv[1] + ".out.missing-b0.45.txt", "w")

for k1 in sorted([2.6, 2.5]):
    for b in sorted([0.45]):
#for k1 in np.arange(19.5, 100, 0.5):
#     for b in np.arange(-1, 1.2, 0.1):
         #for k3 in np.arange(100*itt, 100*(itt+1), 10):
         k3 = 0.0
         bm25ranker.update_weights(np.array([k1,k3,b]))
         print >> fh, "k1:%f k3:%f b:%f score:%f" % (k1, k3, b, evaluation.evaluate_all(bm25ranker, queries))
fh.close()
