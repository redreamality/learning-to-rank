import sys
import include
import evaluation, query, ranker
import numpy as np
from ranker.AbstractRankingFunction import AbstractRankingFunction
w = [0]*64
w[int(sys.argv[2])] = 1
wstr = ",".join([str(x) for x in w])

evaluation = evaluation.NdcgEval()
bm25ranker = AbstractRankingFunction(["ranker.model.Linear"], 'first', 64, init=wstr, sample="utils.sample_fixed")
queries = query.load_queries(sys.argv[1], 64)
print evaluation.evaluate_all(bm25ranker, queries)
