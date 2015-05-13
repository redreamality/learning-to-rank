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
from random import randint, choice

from AbstractInterleavedComparison import AbstractInterleavedComparison


class TeamDraftMultileave(AbstractInterleavedComparison):
    """Baseline team draft method."""

    def __init__(self, arg_str=None):
        pass

    def interleave(self, rankers, query, length):
        """updated to match the original method"""
        rankings = []
        for r in rankers:
            r.init_ranking(query)
            rankings.append(r.docids)
        self.nrrankers = len(rankers)
        length = min(min([len(r) for r in rankings]), length)
        # start with empty document list and assignments
        l = []
        lassignments = []
        # determine overlap in top results
        index = 0
        for i in range(length):
            if len(set([r[i] for r in rankings])) == 1:
                l.append(rankings[0][i])
                lassignments.append(-1)
                index += 1
            else:
                break

        indexes = [index] * len(rankings)

        assignments = [0] * len(rankings)

        while len(l) < length:
            minassignment = min(assignments)
            minassigned = [i for i, a in enumerate(assignments)
                           if a == minassignment]

            rindex = choice(minassigned)
            assignments[rindex] += 1
            lassignments.append(rindex)

            while True:
                next_doc = rankings[rindex][indexes[rindex]]
                indexes[rindex] += 1
                if next_doc not in l:
                    l.append(next_doc)
                    break

        return (asarray(l), asarray(lassignments))

    def infer_outcome(self, l, a, c, query):
        """assign clicks for contributed documents"""

        credits = []
        for r in range(self.nrrankers):
            credit = sum([1 if val_a == r and val_c == 1 else 0
                     for val_a, val_c in zip(list(a), list(c))])
            credits.append(credit)
        return credits
