import sys
import include
import evaluation, query, ranker
from ranker.AbstractRankingFunction import AbstractRankingFunction

evaluation = evaluation.NdcgEval()
bm25ranker = AbstractRankingFunction(["ranker.model.BM25"], 'first', 3, sample="utils.sample_fixed")
queries = query.load_queries(sys.argv[1], 64)
print evaluation.evaluate_all(bm25ranker, queries)

# for k1 in np.arange(1.5, 3.5, 0.2):
#     for b in np.arange(0, 1.6, 0.2):
#         bm25ranker.updateweights(np.array([k1,0,b]))
#         print "k1 = "+k1+" b = " + b + " score = " + evaluation.evaluate_all(bm25ranker, queries)