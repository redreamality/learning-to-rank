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
from AbstractInterleavedComparison import AbstractInterleavedComparison

try:
    import gurobipy
except:
    pass


class OptimizedMultileave(AbstractInterleavedComparison):
    """
    An implementation of Optimized Interleave as described in:

    @see: Radlinski, F., & Craswell, N. (2013, February). Optimized
    interleaving for online retrieval evaluation. In Proceedings of the sixth
    ACM international conference on Web search and data mining (pp. 245-254).

    @author: Anne Schuth
    @contact: anne.schuth@uva.nl
    @since: February 2013
    @requires: Gurobi from http://www.gurobi.com/
    """

    def __init__(self, arg_str=None):
        self.verbose = False
        self.credit = getattr(self, "binary_credit")
        if not arg_str is None:
            parser = argparse.ArgumentParser(description=self.__doc__,
                                             prog=self.__class__.__name__)
            parser.add_argument("-c", "--credit", choices=["linear_credit",
                                                           "binary_credit",
                                                           "inverse_credit"],
                                required=True, default="binary_credit")
            args = vars(parser.parse_known_args(split_arg_str(arg_str))[0])
            self.credit = getattr(self, args["credit"])

    def f(self, i):
        # Implemented as footnote 4 suggests
        return 1. / i

    def rank(self, li, R):
        # Implemented as in d'.1
        if li in R:
            return R[li]
        return len(R) + 1

    def aggregate_credit(self, C, clicked=None):
        c = np.zeros(len(C))
        for k in range(len(C)):
            if clicked != None and clicked[k] != 1:
                continue
            c += ((C[k] + len(C) / 2) / len(C)) * (1.0 / k)
        return c

    def binary_credit(self, li, ranks):
        indexes = [index for _, index in sorted([(self.rank(li, ranks), i)
                                         for i in range(len(ranks))])]
        scores = np.array(range(len(ranks))) - (len(ranks) / 2)
        credit = [scores[indexes.index(i)] for i in range(len(ranks))]
        return credit

#    def linear_credit(self, li, rankA, rankB):
#        # Equation (14)
#        return self.rank(li, rankA) - self.rank(li, rankB)
#
#    def inverse_credit(self, li, rankA, rankB):
#        # Equation (15)
#        return 1. / self.rank(li, rankB) - 1. / self.rank(li, rankA)

    def interleave(self, rankers, query, length):
        rankings = []
        for r in rankers:
            r.init_ranking(query)
            rankings.append(r.docids[:length])
        length = min(min([len(r) for r in rankings]), length)

        currentlevel = [([], [0] * len(rankings))]
        nextlevel = []
        for _ in range(length):
            for prefix, indexes in currentlevel:
                addedthislevel = []
                for i in range(len(rankings)):
                    index = indexes[i]
                    ranking = rankings[i]
                    d = None
                    if index < len(ranking):
                        d = ranking[index]
                        while d in prefix:
                            d = None
                            index += 1
                            if index < len(ranking):
                                d = ranking[index]
                            else:
                                break
                        if  d in addedthislevel:
                            continue
                        if d != None:
                            addedthislevel.append(d)
                            branchindexes = indexes[:]
                            branchindexes[i] = index + 1
                            branch = (prefix + [d], branchindexes)
                            nextlevel.append(branch)

            currentlevel = nextlevel
            nextlevel = []

        # L contains allowed multileavings, according to equation (5)
        L = [n for n, _ in currentlevel]
        del currentlevel
        del nextlevel

        # Pre-compute credit for each list l in L
        ranks = []
        for ranking in rankings:
            ranks.append({})
            for i in range(len(ranking)):
                ranks[-1][ranking[i]] = i + 1

        C = [[self.credit(li, ranks) for li in l] for l in L]

        # Construct a set of constraints
        m = gurobipy.Model("system")
        m.params.outputFlag = 0
        P = []
        # Add a parameter Pi for each list that adheres to equation (6)
        for i in range(len(L)):
            P.append(m.addVar(lb=0.0, ub=1.0, name='p%d' % i))
        m.update()
        # Constraint for equation (7)
        m.addConstr(gurobipy.quicksum(P) == 1, 'sum')
        # Constraints for equation(8) for each k
        for k in range(length):
            m.addConstr(np.sum([P[n] * np.sum([C[n][i]
                                               for i in
                                               range(k)],
                                              axis=0)
                                for n in range(len(L))],
                               axis=0) == 0.0,
                        "c%d" % k)

        # Add sensitivity as an objective to the optimization, equation (13)
        S = []

        for i in range(len(L)):
            # Attempts at replacing Equation (9, 10, 11)
            # TODO: this is really not thought through yet
            s = np.product(self.aggregate_credit(C[i]))
            S.append(P[i] * s)

        m.setObjective(gurobipy.quicksum(S), gurobipy.GRB.MAXIMIZE)

        # Optimize the system and if it is infeasible, relax the constraints
        m.optimize()
        if m.status == gurobipy.GRB.INFEASIBLE:
            m.feasRelaxS(1, False, False, True)
            m.optimize()

        if self.verbose:
            for i in range(len(L)):
                print L[i], C[i], P[i].x
            m.printStats()

        # Sample a list l from L using the computed probabilities
        problist = sorted([(P[i].x, L[i], C[i])
                           for i in range(len(L)) if P[i].x > 0])
        cumprob = 0.0
        randsample = random.random()
        for (p, l, C) in problist:
            cumprob += p
            if randsample <= cumprob:
                break
        return (np.asarray(l), C)

    def infer_outcome(self, l, C, clicked, query):
        agg = self.aggregate_credit(C, clicked)
        ranking = [x for (_, x) in sorted([(agg[i], i)
                                           for i in range(len(agg))])]
        return ranking

if __name__ == '__main__':
    class TestRanker:
        def __init__(self, docids):
            self.docids = docids

        def init_ranking(self, query):
            pass

    r1 = TestRanker(["a", "b", "c", "d"])
    r2 = TestRanker(["b", "d", "c", "a"])

    comparison = OptimizedMultileave()
    l, C = comparison.interleave([r1, r2], None, 4)
    outcome = comparison.infer_outcome(l, C, [0, 1, 0, 0], None)
    print outcome