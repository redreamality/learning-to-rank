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

# KH, 2012/08/15

import argparse
import logging

from numpy import asarray, where

from .AbstractHistInterleavedComparison import AbstractHistInterleavedComparison
from .ProbabilisticInterleave import ProbabilisticInterleave
from ..utils import split_arg_str


class HistProbabilisticInterleave(AbstractHistInterleavedComparison):
    """Probabilistic interleaving using historical data"""

    def __init__(self, arg_str=None):
        self.pi = ProbabilisticInterleave(arg_str)
        self.biased = False
        self.marginalize = True
        if arg_str:
            parser = argparse.ArgumentParser(description="Parse arguments for "
                "interleaving method.", prog=self.__class__.__name__)
            parser.add_argument("-b", "--biased")
            parser.add_argument("-m", "--marginalize")
            args = vars(parser.parse_known_args(split_arg_str(arg_str))[0])
            if args["biased"] == "False" or args["biased"] == None \
                or args["biased"] == 0:
                self.biased = False
            else:
                self.biased = True
            if args["marginalize"] == "False" or args["marginalize"] == 0:
                self.marginalize = False
            else:
                self.marginalize = True

    def infer_outcome(self, l, source_context, c, target_r1, target_r2, query):
        # for prob interleave, a = (a, r1, r2)
        (a, r1, r2) = source_context
        if self.marginalize:
            return self._infer_outcome_with_marginalization(l, a, c, r1, r2,
                target_r1, target_r2, query, self.biased)
        else:
            return self._infer_outcome_no_marginalization(l, a, c, r1, r2,
                target_r1, target_r2, query, self.biased)

    def _infer_outcome_with_marginalization(self, l, a, c, r1, r2, target_r1,
        target_r2, query, biased):
        # get outcome using the target rankers
        target_context = (None, target_r1, target_r2)
        if r1 == r2:
            raise ValueError("r1 and r2 cannot point to the same object.")
        outcome = self.pi.infer_outcome(l, target_context, c, query)
        if outcome == 0:
            return 0
        if biased:
            return outcome
        # if biased is False, compensate for bias using importance sampling
        target_p_list = self.pi.get_probability_of_list(l, target_context,
            query)
        orig_context = (None, r1, r2)
        orig_p_list = self.pi.get_probability_of_list(l, orig_context, query)
        if target_p_list == 0 or orig_p_list == 0:
            logging.warn("Encountered zero probabilities: p(l_target) = %.2f, "
                "p(l_orig) = %.2f" % (target_p_list, orig_p_list))
            return 0
        return outcome * target_p_list / orig_p_list

    def _infer_outcome_no_marginalization(self, l, a, c, r1, r2, target_r1,
        target_r2, query, biased):
        # are there any clicks? (otherwise it's a tie)
        click_ids = where(asarray(c) == 1)
        if not len(click_ids[0]):  # no clicks, will be a tie
            return 0
        # for the observed list and assignment, get the outcome (like TD)
        c1 = sum([1. if val_a == 0 and val_c == 1 else .0
            for val_a, val_c in zip(a, c)])
        c2 = sum([1. if val_a == 1 and val_c == 1 else .0
            for val_a, val_c in zip(a, c)])
        outcome = -1. if c1 > c2 else 1 if c2 > c1 else .0
        if biased:
            return outcome
        # if biased is False, compensate for bias using importance sampling
        # get the probability of observing this list and assignment under
        # target and source distribution
        target_p = self._get_probability_of_list_and_assignment(l, a,
            target_r1, target_r2, query)
        if target_p == 0:
            return .0
        orig_p = self._get_probability_of_list_and_assignment(l, a, r1, r2,
            query)
        if orig_p == 0:
            return .0
        r2.init_ranking(query)

        return outcome * target_p / orig_p

    def _get_probability_of_list_and_assignment(self, l, a, r1, r2, query):
        # P(l) = \prod_{doc in result_list} 1/2 P_1(doc) + 1/2 P_2(doc)
        p_l_a = 1.0
        r1.init_ranking(query)
        r2.init_ranking(query)
        for i, doc in enumerate(l):
            if a[i] == -1:
                p_d = r1.get_document_probability(doc)
            elif a[i] == 0:
                p_d = r1.get_document_probability(doc)
            elif a[i] == 1:
                p_d = r2.get_document_probability(doc)
            else:
                logging.warn("Illegal assignment: ", a)
                return .0
            p_l_a *= 0.5 * p_d
            r1.rm_document(doc)
            r2.rm_document(doc)
        return p_l_a
