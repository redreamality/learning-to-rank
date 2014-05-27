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
import time

from .AbstractInterleavedComparison import AbstractInterleavedComparison
from ..utils import split_arg_str


# Maximum number of documents (in A and B) that we can feed to OI* alggorithms
MAX_NUMBER_OF_DOCS = 20

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
                                                       "inverse_credit",
                                                       "negative_credit"],
                            default="linear_credit")
        parser.add_argument("--allowed_leavings",
                            choices=["prefix_constraint",
                                     "sample_prefix_constraint",
                                     "sample_prefix_constraint_constructive"],
                            default="prefix_constraint")
        parser.add_argument("--sample_size", type=int, default=-1)
        parser.add_argument("--prefix_bound", type=int, default=-1)
        parser.add_argument("--verbose", action="store_true", default=False)
        args = vars(parser.parse_known_args(split_arg_str(arg_str))[0])
        self.credit = getattr(self, args["credit"])
        self.allowed_leavings = getattr(self, args["allowed_leavings"])
        self.verbose = args["verbose"]
        self.prefix_bound = args["prefix_bound"]
        self.sample_size = args["sample_size"]

    def f(self, i):
        # Implemented as footnote 4 suggests
        return 1. / i

    def precompute_rank(self, R):
        rank = {}
        for i in xrange(len(R)):
            rank[R[i]] = i + 1
        return rank

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
        L = []
        while len(L) == 0 and prefix_bound <= length:
            L = self.prefix_constraint_bound(rankings, length, prefix_bound)
            prefix_bound += 1
        return L

    def perm_given_index(self, alist, apermindex):
        """
        See http://stackoverflow.com/questions/5602488/random-picks-from-permutation-generator
        """
        alist = alist[:]
        for i in range(len(alist) - 1):
            apermindex, j = divmod(apermindex, len(alist) - i)
            alist[i], alist[i + j] = alist[i + j], alist[i]
        return alist

    def sample(self, docs, length):
        r = random.randint(0, math.factorial(len(docs)))
        l = self.perm_given_index(docs, r)
        l = l[:length]
        return l

    def reject(self, l, rankings):
        indexes = [0] * len(rankings)

        def update(i, l, k):
            if rankings[i][indexes[i]] == l[k]:
                indexes[i] += 1
                while k >= indexes[i] and len(rankings[i]) > indexes[i]:
                    found = False
                    for m in range(k):
                        if rankings[i][indexes[i]] == l[m]:
                            indexes[i] += 1
                            found = True
                            break
                    if not found:
                        break
                return True
            return False

        for k in range(len(l)):
            found = False
            for i in range(len(rankings)):
                if update(i, l, k):
                    found = True
            if not found:
                return True
        return False

    def sample_prefix_constraint(self, rankings, length):
        docs = list(set().union(*rankings))
        L = []
        start = time.time()
        reject1 = reject2 = reject3 = 0
        rejects = {}
        accepts = {}
        while len(L) < 1000 and time.time() < start + 3:
            l = self.sample(docs, length)
            tl = tuple(l)
            if tl in accepts:
                reject1 += 1
                continue
            if tl in rejects:
                reject2 += 1
                continue
            if self.reject(l, rankings):
                reject3 += 1
                rejects[tl] = 1
                continue
            accepts[tl] = 1
            L.append(l)
        if self.verbose:
            print "perms", math.factorial(len(docs))
            print "reject1", reject1
            print "reject2", reject2
            print "reject3", reject3
            print "l", len(L)
            print "time", time.time() - start
        return L

    def sample_prefix_constraint_constructive(self, rankings, length):
        L = []
        start = time.time()
        while len(L) < self.sample_size and time.time() < start + 10:
            l = []
            indexes = [0] * len(rankings)
            while len(l) < length:
                r = random.randint(0, len(rankings) - 1)
                if indexes[r] >= length:
                    continue
                d = rankings[r][indexes[r]]
                indexes[r] += 1
                while d in l and indexes[r] < length:
                    d = rankings[r][indexes[r]]
                    indexes[r] += 1
                if not d in l:
                    l.append(d)
            if not l in L:
                L.append(l)
        return L

    def interleave(self, r1, r2, query, length, bias=0):
        return self.interleave_n(r1, r2, query, length, 1, bias=0)[0]

    def interleave_n(self, r1, r2, query, length, num_repeat, bias=0):
        import gurobipy

        r1.init_ranking(query)
        r2.init_ranking(query)
        rA, rB = (r.getDocs() for r in [r1, r2])
        length = min(len(rA), len(rB), length)
        assert length <= MAX_NUMBER_OF_DOCS
        # We may need longer rA and rB for interleaving, so this is not a bug.
        rA = rA[:length]
        rB = rB[:length]
        L = self.allowed_leavings([rA, rB], length)
        assert len(L) > 0, (rA, rB, length)

        # Pre-compute credit for each list l in L
        rankA = self.precompute_rank(rA)
        rankB = self.precompute_rank(rB)
        credit = [[self.credit(li, rankA, rankB) for li in l] for l in L]

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
        for k in xrange(length):
            m.addConstr(gurobipy.quicksum([P[n] * credit[n][k] for n in xrange(len(L))]) == 0, "c%d" % k)

        # Add sensitivity as an objective to the optimization, equation (13)
        S = []
        for i in range(len(L)):
            # Equation (9)
            wa = sum([self.f(k + 1)
                      for k in range(len(L[i])) if credit[i][k] > 0])
            # Equation (10)
            wb = sum([self.f(k + 1)
                      for k in range(len(L[i])) if credit[i][k] < 0])
            # Equation (11)
            wt = sum([self.f(k + 1)
                      for k in range(len(L[i])) if credit[i][k] == 0])
            if wa + wb > 0:
                s = -((1 - wt) /
                      (wa + wb)) * math.log(((wa ** wa) * (wb ** wb)) /
                                             ((wa + wb) ** (wa + wb)), 2)
                S.append(P[i] * s)
        m.setObjective(gurobipy.quicksum(S), gurobipy.GRB.MAXIMIZE)

        # Optimize the system and if it is infeasible, relax the constraints
        self.relaxed = False
        m.optimize()
        if m.status == gurobipy.GRB.INFEASIBLE:
            self.relaxed = True
            m.feasRelaxS(1, False, False, True)
            # Restore constraint for equation (7)
            m.addConstr(gurobipy.quicksum(P) == 1, 'sum')
            m.addConstr(gurobipy.quicksum(
                [gurobipy.quicksum([P[n] * credit[n][k] for n in xrange(len(L))]) for k in xrange(length)]
            ) == 0, "c%d" % k)

            m.optimize()

        assert m.status != gurobipy.GRB.INFEASIBLE

        if self.verbose:
            print rA
            print rB
            for i in range(len(L)):
                print L[i], credit[i], P[i].x
            m.printStats()

        # Sample n lists from L using the computed probabilities
        problist = sorted([(P[i].x, L[i], credit[i])
                           for i in range(len(L)) if P[i].x > 0])
        result = []
        for i in xrange(num_repeat):
            cumprob = 0.0
            randsample = random.random()
            for (p, l, cr) in problist:
                cumprob += p
                if randsample <= cumprob:
                    result.append((np.asarray(l), cr))
                    break
        assert len(result) == num_repeat
        return result

    def infer_outcome(self, l, credit, clicks, query):
        return sum(cr for (cr, c) in zip(credit, clicks) if c > 0)
