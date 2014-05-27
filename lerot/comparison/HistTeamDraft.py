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

# KH, 2012/08/14

from random import randint

from .AbstractHistInterleavedComparison import AbstractHistInterleavedComparison


class HistTeamDraft(AbstractHistInterleavedComparison):
    """Team draft method, applied to historical data."""

    def __init__(self, arg_str=None):
        pass

    def _get_possible_assignment(self, l, r1, r2, query):
        r1.init_ranking(query)
        r2.init_ranking(query)
        length = min(r1.document_count(), r2.document_count(), len(l))
        a = []
        # get ranked list for each ranker
        l1, l2 = [], []
        for i in range(length):
            l1.append(r1.next())
            l2.append(r2.next())

        # determine overlap in top results (and whether the overlap matches l)
        i1, i2 = 0, 0
        for i in range(length):
            if l1[i] == l2[i]:
                if l[i] == l1[i]:
                    a.append(-1)
                    i1 += 1
                    i2 += 1
                else:
                    return None
            else:
                break
        # now check pairwise; per rank pair, one document needs to come from
        # each ranker
        while len(a) < length:
            # forward i1 and i2 to point to the next documents not yet in
            # l[0:len(a)]
            while i1 < len(a):
                if l1[i1] in l[:len(a)]:
                    i1 += 1
                else:
                    break
            while i2 < len(a):
                if l2[i2] in l[:len(a)]:
                    i2 += 1
                else:
                    break
            # if there is only one document left, we're fine with a document
            # from either list
            if length - len(a) == 1:
                next_doc = l[len(a)]
                if l1[i1] == next_doc and l2[i2] == next_doc:
                    random_pick = randint(0, 1)
                    a.append(random_pick)
                    if random_pick == 0:
                        i1 += 1
                    else:
                        i2 += 1
                elif l1[i1] == next_doc:
                    a.append(0)
                    i1 += 1
                elif l2[i2] == next_doc:
                    a.append(1)
                    i2 += 1
                else:
                    return None
            else:
                next_1 = l[len(a)]
                next_2 = l[len(a) + 1]
                assert(next_1 != next_2)
                match_1, match_2 = False, False
                # we have a match if the next document matches l1, and the next
                # document from l2 that is not yet in l matches next_2
                if l1[i1] == next_1 and ((l2[i2] == next_1 and
                    l2[i2 + 1] == next_2) or (l2[i2] != next_1 and
                    l2[i2] == next_2)):
                    match_1 = True
                # or if the same is true for l2
                if l2[i2] == next_1 and ((l1[i1] == next_1 and
                    l1[i1 + 1] == next_2) or (l1[i1] != next_1 and
                    l1[i1] == next_2)):
                    match_2 = True
                # two matches: delete one at random
                if match_1 and match_2:
                    if randint(0, 1):
                        match_2 = False
                    else:
                        match_1 = False
                # now we have at most one match left
                if match_1:
                    a.append(0)
                    a.append(1)
                elif match_2:
                    a.append(1)
                    a.append(0)
                else:
                    return None
        return a

    def infer_outcome(self, l, a, c, target_r1, target_r2, query):
        """assign clicks for contributed documents"""

        a = self._get_possible_assignment(l, target_r1, target_r2, query)
        if a == None:
            return 0

        c1 = sum([1 if val_a == 0 and val_c == 1 else 0
            for val_a, val_c in zip(a, c)])
        c2 = sum([1 if val_a == 1 and val_c == 1 else 0
            for val_a, val_c in zip(a, c)])
        return -1 if c1 > c2 else 1 if c2 > c1 else 0
