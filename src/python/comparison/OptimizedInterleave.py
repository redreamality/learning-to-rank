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
from AbstractInterleavedComparison import AbstractInterleavedComparison

try:
    import gurobipy
except:
    pass


class OptimizedInterleave(AbstractInterleavedComparison):
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

    def __init__(self, arg_str=""):
        parser = argparse.ArgumentParser(description=self.__doc__,
                                         prog=self.__class__.__name__)
        parser.add_argument("-c", "--credit", choices=["linear_credit",
                                                       "binary_credit",
                                                       "inverse_credit"],
                            default="linear_credit")
        parser.add_argument("--prefix_bound", type=int, default=-1)
        parser.add_argument("--verbose", action="store_true", default=False)
        args = vars(parser.parse_known_args(split_arg_str(arg_str))[0])
        self.credit = getattr(self, args["credit"])
        self.verbose = args["verbose"]
        self.prefix_bound = args["prefix_bound"]

    def f(self, i):
        # Implemented as footnote 4 suggests
        return 1. / i

    def rank(self, li, R):
        # Implemented as in d'.1
        if li in R:
            return R[li]
        return len(R) + 1

    def binary_credit(self, li, rankA, rankB):
        if self.rank(li, rankA) < self.rank(li, rankB):
            return 1
        elif self.rank(li, rankA) > self.rank(li, rankB):
            return -1
        return 0

    def linear_credit(self, li, rankA, rankB):
        # Equation (14)
        return self.rank(li, rankA) - self.rank(li, rankB)

    def inverse_credit(self, li, rankA, rankB):
        # Equation (15)
        return 1. / self.rank(li, rankB) - 1. / self.rank(li, rankA)

    def prefix_constraint_bound(self, rankings, length, prefix_bound):
        currentlevel = [([], [0] * len(rankings), 1)]
        nextlevel = []
        for _ in range(length):
            for prefix, indexes, indexk in currentlevel:
                addedthislevel = []
                for i in range(len(rankings)):
                    index = indexes[i]
                    ranking = rankings[i]
                    d = None
                    if index < len(ranking) and index <= indexk + prefix_bound:
                        d = ranking[index]
                        while d in prefix:
                            d = None
                            index += 1
                            if index < len(ranking) and index <= indexk + prefix_bound:
                                d = ranking[index]
                            else:
                                break
                        if d in addedthislevel:
                            continue
                        if d != None:
                            addedthislevel.append(d)
                            branchindexes = indexes[:]
                            branchindexes[i] = index + 1
                            if min(branchindexes) > indexk - prefix_bound:
                                branchindexk = indexk + 1
                            else:
                                branchindexk = indexk
                            branch = (prefix + [d], branchindexes, branchindexk)
                            nextlevel.append(branch)

            currentlevel = nextlevel
            nextlevel = []

        # L contains allowed multileavings, according to equation (5)
        L = [n for n, _, _ in currentlevel]
        del currentlevel
        del nextlevel
        return L

    def prefix_constraint(self, rankings, length):
        prefix_bound = length if self.prefix_bound < 0 else self.prefix_bound
        L = self.prefix_constraint_bound(rankings, length, prefix_bound)
        while len(L) == 0:
            prefix_bound += 1
            if prefix_bound > length:
                break
            L = self.prefix_constraint_bound(rankings, length, prefix_bound)
        return L

    def interleave(self, r1, r2, query, length, bias=0):
        r1.init_ranking(query)
        r2.init_ranking(query)
        rA = r1.docids
        rB = r2.docids
        length = min(len(rA), len(rB), length)
        L = self.prefix_constraint([rA, rB], length)

        # Pre-compute credit for each list l in L
        rankA = {}
        for i in range(len(rA)):
            rankA[rA[i]] = i + 1
        rankB = {}
        for i in range(len(rB)):
            rankB[rB[i]] = i + 1
        C = [[self.credit(li, rankA, rankB) for li in l] for l in L]

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
            m.addConstr(gurobipy.quicksum([P[n] * gurobipy.quicksum([C[n][i]
                                                                     for i in
                                                                     range(k)])
                                           for n in
                                           range(len(L))]) == bias,
                        "c%d" % k)

        # Add sensitivity as an objective to the optimization, equation (13)
        S = []
        for i in range(len(L)):
            # Equation (9)
            wa = sum([self.f(k + 1)
                      for k in range(len(L[i])) if C[i][k] > 0])
            # Equation (10)
            wb = sum([self.f(k + 1)
                      for k in range(len(L[i])) if C[i][k] < 0])
            # Equation (11)
            wt = sum([self.f(k + 1)
                      for k in range(len(L[i])) if C[i][k] == 0])
            if wa + wb > 0:
                s = -((1 - wt) /
                      (wa + wb)) * math.log(((wa ** wa) * (wb ** wb)) /
                                             ((wa + wb) ** (wa + wb)), 2)
                S.append(P[i] * s)
        m.setObjective(gurobipy.quicksum(S), gurobipy.GRB.MAXIMIZE)

        # Optimize the system and if it is infeasible, relax the constraints
        m.optimize()
        if m.status == gurobipy.GRB.INFEASIBLE:
            m.feasRelaxS(1, False, True, True)
            m.optimize()

        if self.verbose:
            print rA
            print rB
            for i in range(len(L)):
                print L[i], C[i], P[i].x
            m.printStats()

        # Sample a list l from L using the computed probabilities
        problist = sorted([(P[i].x, L[i], C[i])
                           for i in range(len(L)) if P[i].x > 0])
        cumprob = 0.0
        randsample = random.random()
        p = l = C = None
        for (p, l, C) in problist:
            cumprob += p
            if randsample <= cumprob:
                return (asarray(l), C)

    def infer_outcome(self, l, a, c, query):
        creditsum = 0
        for clicked in where(c == 1)[0]:
            creditsum += a[clicked]
        return creditsum
