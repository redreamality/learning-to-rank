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

import argparse

from numpy import asarray, where
from random import randint

from .AbstractInterleavedComparison import AbstractInterleavedComparison
from ..utils import split_arg_str


class DocumentConstraints(AbstractInterleavedComparison):
    """Interleave using balanced interleave, compare using document
    constraints."""

    def __init__(self, arg_str="random"):
        if arg_str.startswith("--"):
            parser = argparse.ArgumentParser(description="Parse arguments for "
                "interleaving method.", prog=self.__class__.__name__)
            parser.add_argument("-s", "--startinglist", default="random")
            parser.add_argument("-c", "--constraints", type=int, default=3,
                help="Specify which constraint types should be considered. Pos"
                "sible values: 1 - only infer constraints between clicked and "
                "previous non-clicked documents; 2 - in  addition, infer const"
                "raints with the document immediately following a clicked one,"
                " if it was not clicked; 3: in addition infer constraints \w t"
                "he next document that was not clicked.")
            args = vars(parser.parse_known_args(split_arg_str(arg_str))[0])
            self.startinglist = args["startinglist"]
            self.constraints = args["constraints"]
        else:
            self.startinglist = arg_str
            self.constraints = 3

    def interleave(self, r1, r2, query, length):
        # get ranked list for each ranker (put in assignment var)
        l1, l2 = [], []
        r1.init_ranking(query)
        r2.init_ranking(query)
        length = min(r1.document_count(), r2.document_count(), length)
        for _ in range(length):
            l1.append(r1.next())
            l2.append(r2.next())
        # interleave
        l = []
        i1, i2 = 0, 0

        if self.startinglist == "random":
            # pick starting list at random
            first = randint(0, 1)
        elif self.startinglist == "fixed":
            first = 0
        elif self.startinglist == "0":
            first = 0
        elif self.startinglist == "1":
            first = 1
        else:
            raise Exception("Unknown starting method '%s' for "
                            "comparison method %s." %
                            (self.startinglist, self.__class__.__name__))

        # interleave deterministically
        while len(l) < length:
            if (i1 < i2) or (i1 == i2 and first == 0):
                if l1[i1] not in l:
                    l.append(l1[i1])
                i1 += 1
            else:
                if l2[i2] not in l:
                    l.append(l2[i2])
                i2 += 1
        # for balanced interleave the assignment captures the two original
        # ranked result lists l1 and l2
        return (asarray(l), (asarray(l1), asarray(l2)))

    def check_constraints(self, l, a, click_ids):
        c1, c2 = 0, 0
        for hi in click_ids:
            # now hi is the rank of the clicked document, l[hi] is the docid
            # for each clicked document, get documents above it
            doc_range = range(hi)
            # if we're not at the lowest document, add the document after the
            # clicked one as another potential constraint
            # addtl. constraints for non-clicked documents directly after hi
            if self.constraints > 1 and len(l) > hi + 1:
                doc_range.append(hi + 1)
            for lo in doc_range:
                if lo in click_ids:  # y was clicked as well, no constraint.
                    # Add a potential constraint for the document after that.
                    # (OPTIONAL additional constraint)
                    if self.constraints > 2 and lo > hi and len(l) > lo + 1:
                        doc_range.append(lo + 1)
                    continue
                # if those docs are in the same order in l*, then a constraint
                # is violated
                if len(where(a[0] == l[lo])[0]):
                    # doc we want lower is in top N
                    if not len(where(a[0] == l[hi])[0]) or (
                        where(a[0] == l[lo]) < where(a[0] == l[hi])):
                        # and the other is not in top N or has a lower rank
                        c1 += 1
                if len(where(a[1] == l[lo])[0]):
                    if not len(where(a[1] == l[hi])[0]) or (
                        where(a[1] == l[lo]) < where(a[1] == l[hi])):
                        c2 += 1
        return (c1, c2)

    def infer_outcome(self, l, a, c, query):
        c = asarray(c)
        a = (asarray(a[0]), asarray(a[1]))
        click_ids = where(c == 1)[0]
        if not len(click_ids):  # no clicks, will be a tie
            return 0

        # check for violated constraints
        c1, c2 = self.check_constraints(l, a, click_ids)
        # now we have constraints, not clicks, reverse outcome
        return 1 if c1 > c2 else -1 if c2 > c1 else 0
