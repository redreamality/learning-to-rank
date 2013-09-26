import sys
import include
import evaluation, query, ranker
import numpy as np
from ranker.AbstractRankingFunction import AbstractRankingFunction

evaluation = evaluation.NdcgEval()
bm25ranker = AbstractRankingFunction(["ranker.model.Linear"], 'first', 64, sample="utils.sample_fixed")
w = [0]*64
w[24] = 1
bm25ranker.update_weights(np.array(w))
queries = query.load_queries(sys.argv[1], 64)
print evaluation.evaluate_all(bm25ranker, queries)
