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
from random import random

from .AbstractInterleavedComparison import AbstractInterleavedComparison
from ..utils import split_arg_str


class StochasticBalancedInterleave(AbstractInterleavedComparison):
    """Interleave and compare rankers using the stochastic interleave method
    introduced in Hofmann et al. ECIR'11."""

    def _exploration_rate(self, arg_str):
        x = float(arg_str)
        if x > 0 and x < 1:
            return x

    def __init__(self, arg_str):
        self.biased = False
        if arg_str.startswith("-"):
            parser = argparse.ArgumentParser(description="Parse arguments for "
                "interleaving method.", prog=self.__class__.__name__)
            parser.add_argument("-k", "--exploration_rate",
                type="_exploration_rate", required=True)
            parser.add_argument("-b", "--biased")
            args = vars(parser.parse_known_args(split_arg_str(arg_str))[0])
            if args["biased"] == "False" or args["biased"] == 0:
                self.biased = False
            else:
                self.biased = True
            self.k = args["exploration_rate"]
        else:
            try:
                self.k = float(arg_str)
            except Exception as ex:
                raise Exception("arg_str should be parseable by argparse, or "
                    "contain a single float value (the exploration rate k); "
                    "could not parse arg_str:", ex)

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
        a = []
        i1, i2 = 0, 0
        # randomly pick the list to contribute a document at each rank
        while len(l) < length:
            selected = self._pick_list(1, 2)
            a.append(selected)
            if selected == 1:
                while l1[i1] in l:
                    i1 += 1
                l.append(l1[i1])
            else:
                while l2[i2] in l:
                    i2 += 1
                l.append(l2[i2])
        # for balanced interleave the assignment captures the two original
        # ranked result lists l1 and l2
        return (asarray(l), (asarray(l1), asarray(l2), asarray(a)))

    def _pick_list(self, l1, l2):
        # l1 is assumed to be the exploitative list. It is selected with
        # probability 1 - k
        return l1 if random() > self.k else l2

    def infer_outcome(self, l, a, c, query):
        click_ids = where(c == 1)
        if not len(click_ids[0]):  # no clicks, will be a tie
            return 0
        # lowest click
        c_lowest = click_ids[0][-1]
        # project back into l1 and l2
        click_on_l1 = where(a[0] == l[c_lowest])
        click_on_l2 = where(a[1] == l[c_lowest])
        lowest_click = -1
        if len(click_on_l1[0]) and len(click_on_l2[0]):
            lowest_click = min(click_on_l1[0][0], click_on_l2[0][0])
        elif len(click_on_l1[0]):
            lowest_click = click_on_l1[0][0]
        elif len(click_on_l2[0]):
            lowest_click = click_on_l2[0][0]
        # get number of clicked documents ranked higher or equal to N
        # for both lists
        c1, c2 = 0, 0
        for i in click_ids[0]:
            if where(a[0] == l[i]) <= lowest_click:
                c1 += 1
            if where(a[1] == l[i]) <= lowest_click:
                c2 += 1
        # compensate for bias due to exploration rate
        if self.k != 0.5 and self.biased == False:
            n1 = len(where(a[2] == 1)[0])
            n2 = len(a[2]) - n1
            # avoid division by 0
            n1 = 1 if n1 == 0 else n1
            n2 = 1 if n2 == 0 else n2
            c2 = float(n1) / n2 * c2
        # compare and return outcome
        return -1 if c1 > c2 else 1 if c2 > c1 else 0
