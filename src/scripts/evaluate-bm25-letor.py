import sys
import include
import evaluation, query, ranker
from ranker.AbstractRankingFunction import AbstractRankingFunction

evaluation = evaluation.NdcgEval()
bm25ranker = AbstractRankingFunction(["ranker.model.BM25"], 'first', 3, sample="utils.sample_fixed")
queries = query.load_queries(sys.argv[1], 64)
print evaluation.evaluate_all(bm25ranker, queries)
