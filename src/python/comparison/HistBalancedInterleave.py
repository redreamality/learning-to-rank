# KH, 2012/08/21

from numpy import asarray, where

from AbstractHistInterleavedComparison import AbstractHistInterleavedComparison
from BalancedInterleave import BalancedInterleave


class HistBalancedInterleave(AbstractHistInterleavedComparison):
    """Balanced interleave method, applied to historical data."""

    def __init__(self, arg_str=None):
        self.bi = BalancedInterleave()

    def _get_assignment(self, r1, r2, query, length):
        r1.init_ranking(query)
        r2.init_ranking(query)
        length = min(r1.document_count(), r2.document_count(), length)
        # get ranked list for each ranker
        l1, l2 = [], []
        for _ in range(length):
            l1.append(r1.next())
            l2.append(r2.next())
        return (asarray(l1), asarray(l2))

    def infer_outcome(self, l, a, c, target_r1, target_r2, query):
        """count clicks within the top-k interleaved list"""
        return self.bi.infer_outcome(l, self._get_assignment(target_r1,
            target_r2, query, len(l)), c, query)
