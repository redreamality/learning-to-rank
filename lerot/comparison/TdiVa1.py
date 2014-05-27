# This file is part of Lerot.
#
# Lerot is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Lerot is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Lerot.  If not, see <http://www.gnu.org/licenses/>.

import random
from numpy import asarray

from TeamDraft import TeamDraft


class CannotInterleave(Exception):
    pass


class TdiVa1(TeamDraft):
    """ Algorithm 1 described in
    https://bitbucket.org/varepsilon/cikm2013-interleaving """
    def __init__(self, arg_str=None):
        pass

    @staticmethod
    def sampleSmoothly(a, b, maxVal):
        if a > b:
            a, b = b, a
        if a > 0 and b < maxVal:
            randVal = random.randint(a, b + 1)
            if randVal == b + 1:
                return a - 1 if random.randint(0, 1) == 0 else b + 1
            else:
                return randVal
        elif a == 0 and b == maxVal:
            return random.randint(a, b)
        else:   # a > 0 or b < maxVal
            randVal = random.randint(0, 2 * (b - a) + 2)
            if randVal == 2 * (b - a) + 2:
                return (a - 1) if a > 0 else b + 1
            else:
                return a + randVal // 2

    def interleave(self, r1, r2, query, length):
        r1.init_ranking(query)
        r2.init_ranking(query)
        A, B = (X.getDocs() for X in [r1, r2])
#        assert all(isinstance(d, va.Doc) for d in (A + B)), \
#            "all documents passed to TdiVa1 should be of the type \
#            VerticalAwareInterleave.Doc"
        length = min(len(A), len(B), length)

        sizeA = sum(1 for d in A if d.get_type())
        sizeB = sum(1 for d in B if d.get_type())

        sizeL = self.sampleSmoothly(sizeA, sizeB, length - 1)

        def _addNextDocFrom(X, ranking, insideBlock, afterBlock):
            assert (X is A) or (X is B)
            if insideBlock:
                X_left = [x for x in X if x.get_type() and (x not in
                                                            ranking)]
            elif afterBlock:
                X_left = [x for x in X if not x.get_type() and (x not in
                                                                ranking)]
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
            if doc is not None and doc.get_type():
                insideBlock = True
            if doc is None or sum(1 for x in L if x.get_type()) + 1 == sizeL:
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
