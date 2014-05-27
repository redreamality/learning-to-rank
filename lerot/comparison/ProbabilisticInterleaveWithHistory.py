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

# KH, 2012/07/11

import argparse
import logging

from numpy import mean, var

from .ProbabilisticInterleave import ProbabilisticInterleave
from ..utils import string_to_boolean, split_arg_str


class ProbabilisticInterleaveWithHistory(ProbabilisticInterleave):
    """Probabilistic interleaving that reuses historic data (with
    importance sampling)."""

    def __init__(self, arg_str):
        ProbabilisticInterleave.__init__(self, arg_str)
        # parse arguments
        parser = argparse.ArgumentParser(description="Initialize probabilistic"
            " interleave with history.",
            prog="ProbabilisticInterleaveWithHistory")
        parser.add_argument("-l", "--history_length", type=int, required=True,
            help="Number of historical data points to keep in memory and use "
            "to infer feedback.")
        parser.add_argument("-b", "--biased", default=False,
            help="Set to true if comparison should be biased (i.e., not use"
            "importance sampling).")
        if not arg_str:
            raise(Exception("Comparison arguments missing. " +
                parser.format_usage()))
        args = vars(parser.parse_args(split_arg_str(arg_str)))
        self.history_length = args["history_length"]
        self.biased = string_to_boolean(args["biased"])
        logging.info("Initialized historical data usage to: %r" % self.biased)
        # initialize history
        self.history = []

    def infer_outcome(self, l, context, c, query):
        # infer live outcome
        live_outcome = ProbabilisticInterleave.infer_outcome(self, l, context,
            c, query)
        # The following seems to work only from Python 3 onwards (currently
        # using 2.7).
        #live_outcome = super().infer_outcome(l, context, c, query)
        # For each historic data point, infer outcome under the target rankers
        # and re-weight outcomes using importance sampling
        h_outcomes = []
        for h_item in self.history:
            # use the current context (rankers), but historical list and clicks
            raw_outcome = ProbabilisticInterleave.infer_outcome(self,
                h_item.result_list, context, h_item.clicks, h_item.query)
            # probability of the result list under the target distribution
            p_list_target = self.get_probability_of_list(h_item.result_list,
                context, h_item.query)
            if self.biased:
                weight = 1.0
            else:
                weight = p_list_target / h_item.p_list_source
            h_outcomes.append(raw_outcome * weight)
        # TODO: implement alternatives
        # How to actually combine the two estimates? Supposedly, they
        # are both estimates of the expected value of the comparison outcome
        # under the target distribution (rankers), so we should just be able to
        # average them out? But then, the estimator based on historical data
        # has a much higher variance than the live estimator (in fact,
        # infinitely higher, because we only have one estimate from live data
        # so that the variance is zero)
        combined_outcome = .0
        mean_hist = mean(h_outcomes) if len(h_outcomes) > 0 else .0
        if live_outcome == .0 and mean_hist != .0:
            combined_outcome = mean_hist
        elif live_outcome != .0 and mean_hist == .0:
            combined_outcome = live_outcome
        else:
            var_live = 1.0
            var_hist = var(h_outcomes) if len(h_outcomes) > 1 else 1000.0
            combined_outcome = ((var_live * mean_hist + var_hist *
                                 live_outcome) / (var_live + var_hist))

        # add current live data point to history (and keep below or at length
        # self.history_length)
        if self.history_length > 0:
            if len(self.history) and len(self.history) == self.history_length:
                self.history.pop(0)
            # store probability of the observed list under the source
            # distribution so that it only has to be computed once
            new_h_item = HistoryItem(l, context, c, query)
            new_h_item.p_list_source = self.get_probability_of_list(l, context,
                query)
            self.history.append(new_h_item)

        # return the combined outcome
        return combined_outcome


class HistoryItem:
    """Helper class to store a history item."""

    def __init__(self, result_list, context, clicks, query):
        self.result_list = result_list
        self.context = context
        self.clicks = clicks
        self.query = query
