from .AbstractEval import AbstractEval

class VSEval(AbstractEval):
    """Simple vertical selection (VS) metric, a.k.a. prec_v."""

    def __init__(self):
        pass

    def get_value(self, ranking, labels, orientations, cutoff=-1):
        if cutoff == -1:
            cutoff = len(ranking)
        verts_retrieved = set(d.get_type() for d in ranking[:cutoff] if d.get_type() != 'Web')
        if len(verts_retrieved) == 0:
            return 0.0
        verts_retrieved_relevant = sum(1 for v in verts_retrieved if orientations[v] > 0.5)
        return float(verts_retrieved_relevant) / len(verts_retrieved)
