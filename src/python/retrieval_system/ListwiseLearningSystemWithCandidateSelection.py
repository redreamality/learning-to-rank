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
import logging
import numpy as np
import random
import copy


from ListwiseLearningSystem import ListwiseLearningSystem
from utils import string_to_boolean, split_arg_str


class ListwiseLearningSystemWithCandidateSelection(ListwiseLearningSystem):
    """A retrieval system that learns online from listwise comparisons, and
    pre-selects exploratory rankers using historic data."""

    def __init__(self, feature_count, arg_str):
        """
        @param featur_count: the number of features
        @param arg_str: "-h HISTORY_LENGTH -e NUM_CANDIDATES \
            -s SELECT_CANDIDATE".
        """
        ListwiseLearningSystem.__init__(self, feature_count, arg_str)

        parser = argparse.ArgumentParser(prog=self.__class__.__name__)
        parser.add_argument("-e", "--num_candidates", required=True, type=int,
            help="Number of candidate rankers to explore in each round.")
        parser.add_argument("-l", "--history_length", required=True, type=int,
            help="Number of historic data points to take into account when "
            "pre-selecting candidates.")
        parser.add_argument("-s", "--select_candidate", required=True,
            help="Method for selecting a candidate ranker from a ranker pool."
            " Options: select_candidate_random, select_candidate_simple,"
            " select_candidate_repeated, or own implementation.")
        parser.add_argument("-b", "--biased", default="False",
            help="Set to true if comparison should be biased (i.e., not use"
            "importance sampling).")
        parser.add_argument("-r", "--num_repetitions", type=int, default=1,
            help="The number of repetitions for each ranker pair evaluation"
            "(when the selection method is select_candidate_repeated).")
        args = vars(parser.parse_known_args(split_arg_str(arg_str))[0])

        self.num_candidates = args["num_candidates"]
        self.select_candidate = getattr(self, args["select_candidate"])
        self.history_length = args["history_length"]
        self.biased = string_to_boolean(args["biased"])
        logging.info("Initialized historical data usage to: %r" % self.biased)
        self.num_repetitions = args["num_repetitions"]
        self.history = []

    def _get_candidate(self):
        # TODO: generate NUM_CANDIDATES candidate rankers
        candidates = []
        for _ in range(self.num_candidates):
            candidate_ranker, candidate_u = self._get_new_candidate()
            candidates.append(RankerWithU(candidate_ranker, candidate_u))
        # compare them using the HISTORY_LENGTH most recent data points
        # return the most promising one
        best_ranker = self.select_candidate(candidates)
        return best_ranker.ranker, best_ranker.u

    def select_candidate_random(self, candidates):
        return random.sample(candidates, 1)[0]

    def select_candidate_simple(self, candidates):
        """Selects a ranker in randomized matches. For each historic data point
        two rankers are randomly selected from the pool and compared. If a
        ranker loses the comparison, it is removed from the pool. If there is
        more than one ranker left when the history is exhausted, a ranker is
        randomly selected from the remaining pool. This selection method
        assumes transitivity (a ranker that loses against one ranker is assumed
        to not be the best ranker)."""
        count_history = 0
        for h_item in random.sample(self.history, len(self.history)):
            count_history += 1
            sampled_pair = random.sample(candidates, 2)
            candidate_context = ([], sampled_pair[0].ranker,
                sampled_pair[1].ranker)
            # use the current context (rankers), but historical list and clicks
            raw_outcome = self.comparison.infer_outcome(h_item.result_list,
                candidate_context, h_item.clicks, h_item.query)
            if raw_outcome < 0:
                # first ranker won, remove the other from the candidate pool
                candidates.remove(sampled_pair[1])
            elif raw_outcome > 0:
                # and vice versa
                candidates.remove(sampled_pair[0])
            if (len(candidates) == 1):
                break
        logging.debug("Selecting from %d candidates after %d comparisons." % (
            len(candidates), count_history))
        if len(candidates) == 1:
            return candidates[0]
        else:
            return random.sample(candidates, 1)[0]

    def select_candidate_repeated(self, candidates):
        """Selects a ranker in randomized matches. Ranker pairs are sampled
        uniformly and compared over a number of historical samples. The
        outcomes observed over these samples are averaged (with / without
        importance sampling). The worse-performing ranker is removed from the
        pool. If no preference is found, the ranker to be removed is selected
        randomly. The final ranker in the pool is returned. This selection
        method assumes transitivity."""
        if len(self.history) == 0:
            return random.sample(candidates, 1)[0]
        while len(candidates) > 1:
            sampled_pair = random.sample(candidates, 2)
            candidate_context = ([], sampled_pair[0].ranker,
                sampled_pair[1].ranker)
            outcomes = []
            for _ in range(self.num_repetitions):
                h_item = random.sample(self.history, 1)[0]
                raw_outcome = self.comparison.infer_outcome(h_item.result_list,
                    candidate_context, h_item.clicks, h_item.query)
                if self.biased:
                    outcomes.append(raw_outcome)
                else:
                    p_list_target = self.comparison.get_probability_of_list(
                        h_item.result_list, candidate_context, h_item.query)
                    weight = p_list_target / h_item.p_list_source
                    outcomes.append(raw_outcome * weight)
            mean_outcome = np.mean(outcomes)
            if mean_outcome < 0:
                candidates.remove(sampled_pair[1])
            elif mean_outcome > 0:
                candidates.remove(sampled_pair[0])
            else:
                candidates.remove(random.sample(sampled_pair, 1)[0])
        return candidates[0]

    def select_candidate_beat_the_mean(self, candidate_us):
        raise NotImplementedError()

    def _update_solution(self, outcome, clicks):
        # Keep track of history
        if self.history_length > 0:
            if len(self.history) == self.history_length:
                self.history.pop(0)
            # store probability of the observed list under the source
            # distribution so that it only has to be computed once
            new_h_item = HistoryItem(self.current_l, self.current_context,
                clicks, self.current_query)
            new_h_item.p_list_source = self.comparison.get_probability_of_list(
                self.current_l, self.current_context, self.current_query)
            self.history.append(new_h_item)
        # use inherited method for the actual update
        return ListwiseLearningSystem._update_solution(self, outcome, clicks)


class RankerWithU:
    """Helper class to store a ranker and the vector u used to generate it."""
    def __init__(self, ranker, u):
        self.ranker = ranker
        self.u = u


class HistoryItem:
    """Helper class to store a history item."""

    def __init__(self, result_list, context, clicks, query):
        self.result_list = result_list
        self.context = context
        self.clicks = clicks
        self.query = query
