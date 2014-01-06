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
import numpy as np
import math
from utils import split_arg_str
from OptimizedInterleave import OptimizedInterleave
import gurobipy
import os


class OptimizedMultileave(OptimizedInterleave):
    """
    An implementation of Optimized Multileaved inspired by:

    @see: Radlinski, F., & Craswell, N. (2013, February). Optimized
    interleaving for online retrieval evaluation. In Proceedings of the sixth
    ACM international conference on Web search and data mining (pp. 245-254).

    @author: Anne Schuth
    @contact: anne.schuth@uva.nl
    @since: December 2013
    @requires: Gurobi from http://www.gurobi.com/
    """

    def __init__(self, arg_str=""):
        OptimizedInterleave.__init__(self, arg_str)
        parser = argparse.ArgumentParser(description=self.__doc__,
                                         prog=self.__class__.__name__)
        parser.add_argument("-c", "--credit", choices=["inverse_credit",
                                                       "negative_credit"],
                            default="inverse_credit")
        parser.add_argument("--bias", choices=["per_k_bias", "position_bias"],
                            default="per_k_bias")
        parser.add_argument("--sensitivity", choices=["Floor", "Shimon"],
                            default="Floor")
        args = vars(parser.parse_known_args(split_arg_str(arg_str))[0])
        self.credit = getattr(self, args["credit"])
        self.bias = args["bias"]
        self.sensitivity = args["sensitivity"]

    def f(self, i):
        # Implemented as footnote 4 suggests
        return 1. / i

    def inverse_credit(self, li, ranking):
        rank = len(ranking) + 1
        if li in ranking:
            rank = ranking.index(li) + 1
        return 1.0 / rank

    def negative_credit(self, li, ranking):
        rank = len(ranking) + 1
        if li in ranking:
            rank = ranking.index(li) + 1
        return -rank

    def interleave(self, rankers, query, length):
        rankings = []
        for r in rankers:
            r.init_ranking(query)
            rankings.append(r.docids)
        length = min(min([len(r) for r in rankings]), length)
        L = self.allowed_leavings(rankings, length)
        print len(L)

        # Pre-compute credit for each list l in L
        C = [[[self.credit(li, ranking)
               for ranking in rankings]
              for li in l]
             for l in L]

        # Construct a set of constraints
        m = gurobipy.Model("system")
        m.params.outputFlag = 0
        P = []
        # Add a parameter Pi for each list that adheres to equation (6)
        for i in range(len(L)):
            P.append(m.addVar(lb=0.0, ub=1.0, name='p%d' % i))
        m.update()
        sumconstr = m.addConstr(gurobipy.quicksum(P) == 1, 'sum')
        biasconstrs = []
        if self.bias == "per_k_bias":
            V = []
            for k in range(length):
                V.append(m.addVar(name='var%d' % k))
            m.update()
            # Constraint for equation (7)
            # Constraints for equation(8) for each k
            for k in range(length):
                for x in range(len(rankings)):
                    s = []
                    for i in range(len(L)):
                        s.append(P[i] * gurobipy.quicksum(
                                            [C[i][j][x] for j in range(k)]))
                    biasconstrs.append(
                        m.addConstr(gurobipy.quicksum(s) == V[k], "c%d" % k))
        elif self.bias == "position_bias":
            V = [m.addVar(name='var')]
            m.update()
            for x in range(len(rankings)):
                s = []
                for i in range(len(L)):
                    s.append(P[i] * gurobipy.quicksum(
                                            [self.f(j + 1) * C[i][j][x]
                                             for j in range(length)]))
                biasconstrs.append(
                        m.addConstr(gurobipy.quicksum(s) == V[0], "c%d" % x))

        # Add sensitivity as an objective to the optimization, equation (13)
        S = []
        for i in range(len(L)):
            # Replacing Equation (9, 10, 11)
            s = []

            if self.sensitivity == "Floor":
                mu = 0.0
                for x in range(len(rankings)):
                    for j in range(length):
                        mu += self.f(j + 1) * C[i][j][x]
                mu /= len(rankings)
                for x in range(len(rankings)):
                    s.append((sum([
                              self.f(j + 1) * C[i][j][x]
                              for j in range(length)]) - mu) ** 2
                             )
            elif self.sensitivity == "Shimon":
                for x in range(len(rankings)):
                    mu = 0.0
                    for j in range(length):
                        mu += self.f(j + 1) * C[i][j][x]
                    mu /= length
                    s.append((sum([
                              self.f(j + 1) * C[i][j][x] - mu
                              for j in range(length)])) ** 2
                             )

            S.append(P[i] * sum(s))

        m.setObjective(gurobipy.quicksum(S), gurobipy.GRB.MINIMIZE)

        # Optimize the system and if it is infeasible, relax the constraints
        m.optimize()
        if m.status == gurobipy.GRB.INFEASIBLE:
            m.feasRelaxS(1, False, True, True)
            m.optimize()

        # Sample a list l from L using the computed probabilities
        sumprob = sum([P[i].x for i in range(len(L)) if P[i].x > 0])
        problist = sorted([(P[i].x / sumprob, L[i], C[i])
                           for i in range(len(L)) if P[i].x > 0])
        if self.verbose:
            m.printStats()
            for (p, l, C) in problist:
                print l, p

        randsample = random.random()
        p = l = C = None
        cumprob = 0.0
        for (p, l, C) in problist:
            cumprob += p
            if randsample <= cumprob:
                return (np.asarray(l), C)

    def infer_outcome(self, l, C, clicked, query):
        creditsum = np.zeros(len(C[0]))
        for c in np.where(np.array(clicked) == 1)[0]:
            creditsum += C[c]
        return creditsum

if __name__ == '__main__':
    class TestRanker:
        def __init__(self, docids):
            self.docids = docids

        def init_ranking(self, query):
            pass

#    r1 = TestRanker(["a", "b", "c", "d"])
#    r2 = TestRanker(["b", "d", "c", "a"])
#    r3 = TestRanker(["z", "y", "c", "d", "b", "a"])
#    r4 = TestRanker(["c", "d", "x", "a", "1"])
#    r5 = TestRanker(["f", "g", "c", "d", "x", "a"])
#
#    rankers = [r1, r2, r3, r4, r5]

    nrrankers = 5
    lenrankers = 10
    rankers = [TestRanker(sorted(range(lenrankers+8), key=os.urandom)[1:lenrankers+1]) for nr in range(nrrankers)]
    print random.shuffle(range(lenrankers))
    for i in range(len(rankers)):
        print "r%d" % i, rankers[i].docids

    comparison = OptimizedMultileave("--allowed_leavings sample_prefix_constraint_constructive")
    l, C = comparison.interleave(rankers, None, 10)
    print "l", l

#    comparison = OptimizedMultileave()
#    counter = {}
#    for i in range(1000):
#        l, C = comparison.interleave(rankers, None, 4)
#        key = tuple(l)
#        if not key in counter:
#            counter[key] = 0
#        counter[key] += 1
#    for l in sorted(counter.keys()):
#        print l, counter[l] / 1000.0
#    clicks = np.zeros(len(l))
#    for rdoc in ["a", "b", "c", "d"]:
#        if rdoc in l:
#            rindex = l.tolist().index(rdoc)
#            clicks[rindex] = 1
#    print "clicks", clicks
#    outcome = comparison.infer_outcome(l, C, clicks, None)
#    print "outcome", outcome
