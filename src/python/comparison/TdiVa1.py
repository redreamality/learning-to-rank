import random
from numpy import asarray

import VerticalAwareInterleave as va
from TdiVa import TdiVa


class CannotInterleave(Exception):
    pass


class TdiVa1(TdiVa):
    """ Algorithm 1 described in
    https://bitbucket.org/varepsilon/cikm2013-interleaving """
    def __init__(self, arg_str=None):
        pass

    def interleave(self, r1, r2, query, length):
        r1.init_ranking(query)
        r2.init_ranking(query)
        A, B = (X.getDocs() for X in [r1, r2])
        assert all(isinstance(d, va.Doc) for d in (A + B)), \
            "all documents passed to TdiVa1 should be of the type \
            VerticalAwareInterleave.Doc"
        length = min(len(A), len(B), length)

        sizeA = sum(1 for d in A if d.vert)
        sizeB = sum(1 for d in B if d.vert)

        sizeL = self.sampleSmoothly(sizeA, sizeB, length - 1)

        def _addNextDocFrom(X, ranking, insideBlock, afterBlock):
            assert (X is A) or (X is B)
            if insideBlock:
                X_left = [x for x in X if x.vert and (x not in ranking)]
            elif afterBlock:
                X_left = [x for x in X if not x.vert and (x not in ranking)]
            else:
                X_left = [x for x in X if x not in ranking]
            if not X_left:
                return None
            else:
                return X_left[0]

        # start with empty document list and assignments
        L, assignment = [], []
        teamA, teamB = 0, 0
        insideBlock = False
        afterBlock = False
        while len(L) < length:
            doc = None
            a = None
            if teamA < teamB + random.randint(0, 1):
                doc = _addNextDocFrom(A, L, insideBlock, afterBlock)
                a = 0
            else:
                doc = _addNextDocFrom(B, L, insideBlock, afterBlock)
                a = 1
            if afterBlock and doc is None:
                raise CannotInterleave('no non-vertical documents left; try \
                with bigger vertical block')
            if doc is not None and doc.vert:
                insideBlock = True
            if doc is None or sum(1 for x in L if x.vert) + 1 == sizeL:
                # the block is built
                insideBlock = False
                afterBlock = True
            if doc is not None:
                L.append(doc)
                assignment.append(a)
                if a == 0:
                    teamA += 1
                else:
                    teamB += 1
        return (asarray(L), assignment)
