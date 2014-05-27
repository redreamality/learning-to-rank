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
import itertools

from numpy import asarray

from .TeamDraft import TeamDraft


class CannotInterleave(Exception):
    pass


NUM_RETRIES = 1000


class VaTdi(TeamDraft):
    """ Algorithm described in
    https://bitbucket.org/varepsilon/tois2013-interleaving """
    def __init__(self, arg_str=None):
        pass

    @staticmethod
    def sampleSmoothly(a, b, maxVal):
        if a > b:
            a, b = b, a
        if b > maxVal:
            b = maxVal
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

    def interleave(self, r1, r2, query, length=None):
        for i in xrange(NUM_RETRIES):
            try:
                return self._interleave(r1, r2, query, length)
            except CannotInterleave as e:
                pass
        raise Exception("Unable to interleave after %d attempts" % NUM_RETRIES)


    def _interleave(self, r1, r2, query, length1=None):
        r1.init_ranking(query)
        r2.init_ranking(query)
        A, B = (X.getDocs() for X in [r1, r2])
        length = min(len(A), len(B))
        if length1 is not None:
            length = min(length, length1)

        allVertTypes = set((d.get_type() for d in itertools.chain(A, B)
                            if d.get_type() != 'Web'))

        # Size of different vertical blocks. Initially set to zero for all verts.
        sizeL = dict(((t, 0) for t in allVertTypes))
        for t in sizeL.iterkeys():
            At = set(d for d in A if d.get_type() == t)
            Bt = set(d for d in B if d.get_type() == t)
            sizeA = len(At)
            sizeB = len(Bt)
            maxSize = min(length, len(At | Bt), 2 * sizeA + 1, 2 * sizeB + 1)
            sizeL[t] = self.sampleSmoothly(sizeA, sizeB, maxSize)

        def _addNextDocFrom(X, ranking, currentVert, vLeft):
            assert (X is A) or (X is B)
            if currentVert != 'Web':
                X_left = [x for x in X if x.get_type() == currentVert \
                          and (x not in ranking)]
            else:
                X_left = [x for x in X if x.get_type() in vLeft and (x not in ranking)]
            if len(X_left) == 0:
                raise CannotInterleave("No more documents of type %s. "
                                       "sizeL = %s, A = %s, B = %s, L = %s" % (
                                           currentVert,
                                           str(sizeL),
                                           str(A), str(B), str(L)))
            return X_left[0]

        # Start with an empty document list and assignments.
        L, assignment = [], []
        teamA, teamB = 0, 0
        # All the verticals are not used in the beginning (except for 0-sized).
        vLeft = set((k for (k, v) in sizeL.iteritems() if v != 0))
        vLeft.add('Web')       # web document should always be available
        currentVert = 'Web'
        while len(L) < length:
            doc = None
            if teamA < teamB + random.randint(0, 1):
                doc = _addNextDocFrom(A, L, currentVert, vLeft)
                assignment.append(0)
                teamA += 1
            else:
                doc = _addNextDocFrom(B, L, currentVert, vLeft)
                assignment.append(1)
                teamB += 1
            L.append(doc)
            currentVert = doc.get_type()
            if currentVert != 'Web' and sum(1 for d in L if \
                    d.get_type() == currentVert) == sizeL[currentVert]:
                vLeft.remove(currentVert)
                currentVert = 'Web'
        return (asarray(L), assignment)
