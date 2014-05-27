from collections import defaultdict

from .AbstractEval import AbstractEval

class ISEval(AbstractEval):
    """Simple vertical selection (IS) metric, a.k.a. mean-prec."""

    def __init__(self):
        pass

    def get_value(self, ranking, labels, orientations, cutoff=-1):
        if cutoff == -1:
            cutoff = len(labels)
        stats_by_vert = defaultdict(lambda: {'total': 0, 'rel': 0})
        for d in ranking[:cutoff]:
            vert = d.get_type()
            if vert == 'Web':
                continue
            stats_by_vert[vert]['total'] += 1
            if labels[d.get_id()] > 0:
                stats_by_vert[vert]['rel'] += 1
        precisions = [float(s['rel']) / s['total'] for s in stats_by_vert.itervalues()]
        if len(precisions) == 0:
            return 0.0
        return float(sum(precisions)) / len(precisions)
