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

# KH, 2012/07/20

import argparse
import random

from ListwiseLearningSystem import ListwiseLearningSystem
from retrieval_system import AbstractOracleSystem
from utils import get_class, split_arg_str


class ListwiseLearningSystemWithOracle(ListwiseLearningSystem,
    AbstractOracleSystem):
    """A retrieval system that learns online from listwise comparisons, and
    pre-selects exploratory rankers using an oracle."""

    def __init__(self, feature_count, arg_str):
        """
        @param featur_count: the number of features
        @param arg_str: "-e NUM_CANDIDATES -m EVALUATION".
        """
        ListwiseLearningSystem.__init__(self, feature_count, arg_str)

        parser = argparse.ArgumentParser(prog=self.__class__.__name__)
        parser.add_argument("-e", "--num_candidates", required=True, type=int,
            help="number of candidate rankers to explore in each round")
        parser.add_argument("-m", "--evaluation", required=True,
            help="evaluation metric for computing oracle performance")
        parser.add_argument("-s", "--sample_size", default=10, type=int,
            help="size of the query sample to use for oracle comparisons")
        args = vars(parser.parse_known_args(split_arg_str(arg_str))[0])

        self.num_candidates = args["num_candidates"]
        self.sample_size = args["sample_size"]
        self.evaluation_class = get_class(args["evaluation"])
        self.evaluation = self.evaluation_class()

    def _get_candidate(self):
        # TODO: generate NUM_CANDIDATES candidate rankers
        candidates = []
        for _ in range(self.num_candidates):
            candidate_u = self.sample(self.feature_count)
            candidate_weights = self.ranker.w + self.delta * candidate_u
            candidate_ranker = self.ranker_class(self.ranker_args,
                candidate_weights, self.ranker_tie)
            candidates.append(RankerWithU(candidate_ranker, candidate_u))
        # compare them using the HISTORY_LENGTH most recent data points
        # return the most promising one
        best_ranker = self.select_candidate_oracle(candidates)
        self.current_u = best_ranker.u

        return self.ranker.w + self.delta * self.current_u

    def select_candidate_oracle(self, candidates):
        scores = []
        query_sample = random.sample(self.test_queries, self.sample_size)
        for candidate in candidates:
            scores.append(self.evaluation.evaluate_all(
                candidate.ranker.w, query_sample))
        sorted_candidates = [candidate for _, candidate in
            sorted(zip(scores, candidates), reverse=True)]
        return sorted_candidates[0]

    def _update_solution(self, outcome, clicks):
        # use inherited method for the actual update
        return ListwiseLearningSystem._update_solution(self, outcome, clicks)


class RankerWithU:
    """Helper class to store a ranker and the vector u used to generate it."""
    def __init__(self, ranker, u):
        self.ranker = ranker
        self.u = u
