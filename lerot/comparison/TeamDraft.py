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

# KH, 2012/06/19

from numpy import asarray
from random import randint

from .AbstractInterleavedComparison import AbstractInterleavedComparison


class TeamDraft(AbstractInterleavedComparison):
    """Baseline team draft method."""

    def __init__(self, arg_str=None):
        pass

    def interleave(self, r1, r2, query, length1=None):
        """updated to match the original method"""
        r1.init_ranking(query)
        r2.init_ranking(query)
        length = min(r1.document_count(), r2.document_count())
        if length1 is not None:
            length = min(length, length1)
        # start with empty document list and assignments
        l, a = [], []
        # get ranked list for each ranker
        l1, l2 = r1.getDocs(length), r2.getDocs(length)
        i1, i2 = 0, 0

        # determine overlap in top results
        for i in range(length):
            if l1[i] == l2[i]:
                l.append(l1[i])
                a.append(-1)
                i1 += 1
                i2 += 1
            else:
                break

        a1, a2 = 0, 0
        while len(l) < length:
            if (a1 < a2) or (a1 == a2 and randint(0, 1) == 0):
                a.append(0)
                a1 += 1
                while True:
                    next_doc = l1[i1]
                    i1 += 1
                    if next_doc not in l:
                        l.append(next_doc)
                        break
            else:
                a.append(1)
                a2 += 1
                while True:
                    next_doc = l2[i2]
                    i2 += 1
                    if next_doc not in l:
                        l.append(next_doc)
                        break
        return (asarray(l), asarray(a))

    def infer_outcome(self, l, a, c, query):
        """assign clicks for contributed documents"""

        c1 = sum([1 if val_a == 0 and val_c == 1 else 0
            for val_a, val_c in zip(a, c)])
        c2 = sum([1 if val_a == 1 and val_c == 1 else 0
            for val_a, val_c in zip(a, c)])
        return -1 if c1 > c2 else 1 if c2 > c1 else 0
