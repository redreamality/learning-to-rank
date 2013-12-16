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
import argparse
from numpy import asarray, where
import math
from utils import split_arg_str
from OptimizedInterleave import OptimizedInterleave

try:
    import gurobipy
except:
    pass


class CascadeOptimizedInterleave(OptimizedInterleave):
    class DummyRanker:
        def __init__(self, docids):
            self.docids = docids

        def init_ranking(self, query):
            pass

    def interleave(self, rankers, query, length):
        r1 = rankers[0]
        bias = 0
        i = 0
        for r2 in rankers[1:]:
            i += 1
            bias = -3
            l, C = OptimizedInterleave.interleave(self, r1, r2, query, length,
                                                  bias=bias)
            r1 = self.DummyRanker(l)
        return l, C


if __name__ == '__main__':
    class TestRanker:
        def __init__(self, docids):
            self.docids = docids

        def init_ranking(self, query):
            pass

    r1 = TestRanker(["a", "b", "c", "d"])
    r2 = TestRanker(["b", "d", "c", "a"])
    r3 = TestRanker(["c", "d", "b", "a"])
    r4 = TestRanker(["c", "d", "x", "a"])

    rankers = [r1, r2, r3, r4]

    for i in range(len(rankers)):
        print "r%d" % i, rankers[i].docids
    comparison = CascadeOptimizedInterleave("--verbose")
    l, C = comparison.interleave(rankers, None, 4)
    print "l", l

    comparison = CascadeOptimizedInterleave()
    counter = {}
    for i in range(1000):
        l, C = comparison.interleave(rankers, None, 4)
        key = tuple(l)
        if not key in counter:
            counter[key] = 0
        counter[key] += 1
    for l in sorted(counter.keys()):
        print l, counter[l] / 1000.0
#    clicks = np.zeros(len(l))
#    for rdoc in ["a", "b", "c", "d"]:
#        if rdoc in l:
#            rindex = l.tolist().index(rdoc)
#            clicks[rindex] = 1
#    print "clicks", clicks
#    outcome = comparison.infer_outcome(l, C, clicks, None)
#    print "outcome", outcome

