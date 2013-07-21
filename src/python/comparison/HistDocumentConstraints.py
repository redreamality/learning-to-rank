# KH, 2012/08/21

from numpy import asarray, where

from AbstractHistInterleavedComparison import AbstractHistInterleavedComparison
from DocumentConstraints import DocumentConstraints


class HistDocumentConstraints(AbstractHistInterleavedComparison):
    """Document constraints method, applied to historical data."""

    def __init__(self, arg_str=None):
        if arg_str:
            self.dc = DocumentConstraints(arg_str)
        else:
            self.dc = DocumentConstraints()

    def infer_outcome(self, l, a, c, target_r1, target_r2, query):
        """count clicks within the top-k interleaved list"""

        c = asarray(c)
        click_ids = where(c == 1)[0]
        if not len(click_ids):  # no clicks, will be a tie
            return 0

        # get ranked list for each ranker
        target_r1.init_ranking(query)
        target_r2.init_ranking(query)
        length = min(target_r1.document_count(), target_r2.document_count(),
            len(l))
        a = ([], [])
        for _ in range(length):
            a[0].append(target_r1.next())
            a[1].append(target_r2.next())
        a = (asarray(a[0]), asarray(a[1]))

        # check for violated constraints
        c1, c2 = self.dc.check_constraints(l, a, click_ids)
        # now we have constraints, not clicks, reverse outcome
        return 1 if c1 > c2 else -1 if c2 > c1 else 0
