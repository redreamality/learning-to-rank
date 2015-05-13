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

# KH, 2012/06/14
"""
Retrieval system implementation for use in learning experiments.
"""

import argparse
from numpy import array
import copy

from .AbstractLearningSystem import AbstractLearningSystem
from ..utils import get_class, split_arg_str


class ListwiseLearningSystem(AbstractLearningSystem):
    """A retrieval system that learns online from listwise comparisons. The
    system keeps track of all necessary state variables (current query,
    weights, etc.) so that comparison and learning classes can be stateless
    (implement only static / class methods)."""

    def __init__(self, feature_count, arg_str):
        # parse arguments
        parser = argparse.ArgumentParser(description="Initialize retrieval "
            "system with the specified feedback and learning mechanism.",
            prog="ListwiseLearningSystem")
        parser.add_argument("-w", "--init_weights", help="Initialization "
            "method for weights (random, zero).", required=True)
        parser.add_argument("--sample_weights", default="sample_unit_sphere")
        parser.add_argument("-c", "--comparison", required=True)
        parser.add_argument("-f", "--comparison_args", nargs="*")
        parser.add_argument("-r", "--ranker", required=True)
        parser.add_argument("-s", "--ranker_args", nargs="*")
        parser.add_argument("-t", "--ranker_tie", default="random")
        parser.add_argument("-d", "--delta", required=True, type=str)
        parser.add_argument("-a", "--alpha", required=True, type=str)
        parser.add_argument("--anneal", type=int, default=0)
        parser.add_argument("--normalize", default="False")
        args = vars(parser.parse_known_args(split_arg_str(arg_str))[0])

        self.ranker_class = get_class(args["ranker"])
        self.ranker_args = args["ranker_args"]
        self.ranker_tie = args["ranker_tie"]
        self.sample_weights = args["sample_weights"]
        self.init_weights = args["init_weights"]
        self.feature_count = feature_count
        self.ranker = self.ranker_class(self.ranker_args,
                                        self.ranker_tie,
                                        self.feature_count,
                                        sample=self.sample_weights,
                                        init=self.init_weights)

        if "," in args["delta"]:
            self.delta = array([float(x) for x in args["delta"].split(",")])
        else:
            self.delta = float(args["delta"])
        if "," in args["alpha"]:
            self.alpha = array([float(x) for x in args["alpha"].split(",")])
        else:
            self.alpha = float(args["alpha"])

        self.anneal = args["anneal"]

        self.comparison_class = get_class(args["comparison"])
        if "comparison_args" in args and args["comparison_args"] != None:
            self.comparison_args = " ".join(args["comparison_args"])
            self.comparison_args = self.comparison_args.strip("\"")
        else:
            self.comparison_args = None
        self.comparison = self.comparison_class(self.comparison_args)
        self.query_count = 0

    def _get_new_candidate(self):
        w, u = self.ranker.get_candidate_weight(self.delta)
        candidate_ranker = copy.deepcopy(self.ranker)
        candidate_ranker.update_weights(w)
        return candidate_ranker, u

    def _get_candidate(self):
        return self._get_new_candidate()

    def get_ranked_list(self, query, getNewCandidate=True):
        self.query_count += 1
        if self.anneal > 0 and self.query_count % self.anneal == 0:
            self.delta /= 2
            self.alpha /= 2

        if getNewCandidate == True:
            self.candidate_ranker, self.current_u = self._get_candidate()
    
        (l, context) = self.comparison.interleave(self.ranker,
                                                  self.candidate_ranker,
                                                  query,
                                                  10)
        self.current_l = l
        self.current_context = context
        self.current_query = query
        return l

    def update_solution(self, clicks):
        outcome = self.comparison.infer_outcome(self.current_l,
                                                self.current_context,
                                                clicks,
                                                self.current_query)
        if outcome > 0:
            self.ranker.update_weights(self.current_u, self.alpha)
        return self.get_solution()

    def get_solution(self):
        return self.ranker
