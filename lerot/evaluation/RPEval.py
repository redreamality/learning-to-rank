import scipy
import scipy.stats

from .AbstractEval import AbstractEval

class RPEval(AbstractEval):
    """Simple vertical selection (RP) metric, a.k.a. corr."""

    def __init__(self):
        pass

    def get_value(self, ranking, labels, orientations, cutoff=-1, ideal_ranking=None):
        assert ideal_ranking is not None
        if cutoff == -1:
            cutoff = len(ranking)
        cutoff = min([cutoff, len(ranking), len(ideal_ranking)])
        this_page_rels = [labels[d.get_id()] for d in ranking[:cutoff]]
        ideal_page_rels = [labels[d.get_id()] for d in ideal_ranking[:cutoff]]
        return scipy.stats.spearmanr(this_page_rels, ideal_page_rels)[0]
