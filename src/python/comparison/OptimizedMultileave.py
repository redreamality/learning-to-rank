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
import gurobipy


class OptimizedMultileave(AbstractInterleavedComparison):
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
        parser = argparse.ArgumentParser(description=self.__doc__,
                                         prog=self.__class__.__name__)
        parser.add_argument("-c", "--credit", choices=["inverse_credit",
                                                       "negative_credit"],
                            default="inverse_credit")
        parser.add_argument("--verbose", action="store_true", default=False)
        args = vars(parser.parse_known_args(split_arg_str(arg_str))[0])
        self.credit = getattr(self, args["credit"])
        self.verbose = args["verbose"]

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
#            rankings.append(r.docids[:length])
            rankings.append(r.docids)
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
        C = [[[self.credit(li, ranking)
               for ranking in rankings]
              for li in l]
             for l in L]

        # Construct a set of constraints
        m = gurobipy.Model("system")
        m.params.outputFlag = 0
        P = []
        V = []
        # Add a parameter Pi for each list that adheres to equation (6)
        for i in range(len(L)):
            P.append(m.addVar(lb=0.0, ub=1.0, name='p%d' % i))
        for k in range(length):
            V.append(m.addVar(name='var%d' % k))
        m.update()
        # Constraint for equation (7)
        m.addConstr(gurobipy.quicksum(P) == 1, 'sum')
        # Constraints for equation(8) for each k
        for k in range(length):
            for x in range(len(rankings)):
                s = []
                for i in range(len(L)):
                    s.append(P[i] * gurobipy.quicksum(
                                            [C[i][j][x] for j in range(k)]))
                m.addConstr(gurobipy.quicksum(s) == V[k], "c%d" % k)

        # Add sensitivity as an objective to the optimization, equation (13)
        S = []
        for i in range(len(L)):
            # Replacing Equation (9, 10, 11)
            s = []
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
            S.append(P[i] * gurobipy.quicksum(s))

        m.setObjective(gurobipy.quicksum(S), gurobipy.GRB.MINIMIZE)

        # Optimize the system and if it is infeasible, relax the constraints
        m.optimize()
        if m.status == gurobipy.GRB.INFEASIBLE:
            m.feasRelaxS(1, False, True, True)
            #m.feasRelaxS(1, False, True, False)
            #m.feasRelaxS(1, False, False, True)
            m.optimize()

        # Sample a list l from L using the computed probabilities
        sumprob = sum([P[i].x for i in range(len(L)) if P[i].x > 0])
        problist = sorted([(P[i].x / sumprob, L[i], C[i])
                           for i in range(len(L)) if P[i].x > 0])
        if self.verbose:
            m.printStats()
            for (p, l, C) in problist:
                print l, p

        cumprob = 0.0
        randsample = random.random()
        p = l = C = None
        for (p, l, C) in problist:
            cumprob += p
            if randsample <= cumprob:
                break
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

    r1 = TestRanker(["a", "b", "c", "d"])
    r2 = TestRanker(["b", "d", "c", "a"])
    r3 = TestRanker(["c", "d", "b", "a"])

    rankers = [r1, r2, r3]
#    rankers = [r1, r2]

    for i in range(len(rankers)):
        print "r%d" % i, rankers[i].docids

    comparison = OptimizedMultileave("--verbose --credit inverse_credit")
    l, C = comparison.interleave(rankers, None, 3)
    print "interleaving", l
    clicks = np.zeros(len(l))
    for rdoc in ["a", "b", "c", "d"]:
        if rdoc in l:
            rindex = l.tolist().index(rdoc)
            clicks[rindex] = 1
    print "clicks", clicks
    outcome = comparison.infer_outcome(l, C, clicks, None)
    print "outcome", outcome
