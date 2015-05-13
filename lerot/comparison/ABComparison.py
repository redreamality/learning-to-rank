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

# KH, 2014/08/17

from numpy import asarray
from random import randint

from .AbstractInterleavedComparison import AbstractInterleavedComparison


class ABComparison(AbstractInterleavedComparison):
    """Simulate AB test."""

    def __init__(self, arg_str=None):
        pass

    def interleave(self, r1, r2, query, length1=None):
        l, a = [], []
        if randint(0, 1) == 0:
            r1.init_ranking(query)
            length = min(r1.document_count(), length1)
            l = r1.getDocs(length)
            a = [0] * length
        else:
            r2.init_ranking(query)
            length = min(r2.document_count(), length1)
            l = r2.getDocs(length)
            a = [1] * length
        return (asarray(l), asarray(a))

    def infer_outcome(self, l, a, c, query):
        """simply re-use team draft code."""

        c1 = sum([1 if val_a == 0 and val_c == 1 else 0
            for val_a, val_c in zip(a, c)])
        c2 = sum([1 if val_a == 1 and val_c == 1 else 0
            for val_a, val_c in zip(a, c)])
        return -1 if c1 > c2 else 1 if c2 > c1 else 0
