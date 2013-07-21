# KH, 2012/06/20

from numpy import log2

from AbstractEval import AbstractEval


class LetorNdcgEval(AbstractEval):
    """Compute NDCG as implemented in the Letor toolkit."""

    def get_dcg(self, labels, cutoff=-1):
        if (cutoff == -1):
            cutoff = len(labels)
        dcg = 0
        # [0:cutoff] returns the labels up to min(len(labels), cutoff)
        for r, label in enumerate(labels[0:cutoff]):
            # use log2(1 + r), to be consistent with the implementation in the
            # letor 4 evaluation tools (and wikipedia, on 6/27/2012), even
            # though this makes discounting slightly inconsistent (indices are
            # zero-based, so using log2(2 + r) would be more consistent)
            if r == 0:
                dcg += 2 ** label - 1
            else:
                dcg += (2 ** label - 1) / log2(1 + r)
        return dcg
