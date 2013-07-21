# KH, 2012/06/14
# Retrieval system implementation for use in learning experiments.

import argparse

from numpy import array, dot, ones, where, zeros
from numpy.linalg import norm
from random import gauss, random

from AbstractLearningSystem import AbstractLearningSystem
from utils import get_class, split_arg_str


class PairwiseLearningSystem(AbstractLearningSystem):
    """A retrieval system that learns online from pairwise comparisons. The
    system keeps track of all necessary state variables (current query,
    weights, etc.)."""

    def __init__(self, feature_count, arg_str):
        self.feature_count = feature_count
        # parse arguments
        parser = argparse.ArgumentParser(description="Initialize retrieval "
            "system with the specified feedback and learning mechanism.",
            prog="PairwiseLearningSystem")
        parser.add_argument("-w", "--init_weights", help="Initialization "
            "method for weights (random, zero, fixed).", required=True)
        parser.add_argument("-e", "--epsilon", required=True, type=float)
        parser.add_argument("-f", "--eta", required=True, type=float)
        parser.add_argument("-l", "--lamb", type=float, default=0.0)
        parser.add_argument("-r", "--ranker", required=True)
        parser.add_argument("-s", "--ranker_args", nargs="*")
        parser.add_argument("-t", "--ranker_tie", default="random")
        args = vars(parser.parse_known_args(split_arg_str(arg_str))[0])
        # initialize weights, comparison method, and learner
        w = self.initialize_weights(args["init_weights"], self.feature_count)
        self.ranker_class = get_class(args["ranker"])
        if "ranker_args" in args and args["ranker_args"] != None:
            self.ranker_args = " ".join(args["ranker_args"])
            self.ranker_args = self.ranker_args.strip("\"")
        else:
            self.ranker_args = None
        self.ranker_tie = args["ranker_tie"]
        self.ranker = self.ranker_class(self.ranker_args, w, self.ranker_tie)
        self.epsilon = args["epsilon"]
        self.eta = args["eta"]
        self.lamb = args["lamb"]

    def initialize_weights(self, method, feature_count):
        if method == "zero":
            return zeros(self.feature_count)
        elif method == "random":
            return self.sample_unit_sphere(self.feature_count) * 0.01
        elif method == "fixed":
            return self.sample_fixed(self.feature_count) * 0.01
        else:
            try:
                weights = array([float(num) for num in method.split(",")])
                if len(weights) != feature_count:
                    raise Exception("List of initial weights does not have the"
                        " expected length (%d, expected $d)." %
                        (len(weights, feature_count)))
                return weights
            except Exception as ex:
                raise Exception("Could not parse weight initialization method:"
                    " %s. Possible values: zero, random, or a comma-separated "
                    "list of float values that indicate specific weight values"
                    ". Error: %s" % (method, ex))

    def sample_unit_sphere(self, n):
        """See http://mathoverflow.net/questions/24688/efficiently-sampling-
        points-uniformly-from-the-surface-of-an-n-sphere"""
        v = zeros(n)
        for i in range(0, n):
            v[i] = gauss(0, 1)
        return v / norm(v)

    def sample_fixed(self, n):
        v = ones(n)
        return v / norm(v)

    def get_ranked_list(self, query):
        # current ranker
        self.ranker.init_ranking(query)
        length = min(self.ranker.document_count(), 10)
        l = []
        for _ in range(length):
            if random() > self.epsilon:
                # exploitative
                l.append(self.ranker.next())
            else:
                # exploratory (next and next_random also remove the doc from r)
                l.append(self.ranker.next_random())
        self.current_l = l
        self.current_query = query
        return l

    def update_solution(self, clicks):
        """"Ranker weights are updated after each observed document pair. This
        means that a pair may have been misranked when the result list was gen-
        erated, but is correctly labeled after an earlier update based on a
        higher-ranked pair from the same list."""
        click_ids = where(clicks == 1)[0]
        if not len(click_ids):  # no clicks, will be a tie
            return self.ranker.w
        # extract pairwise preferences from clicks
        for hi in click_ids:
            # now hi is the rank of the clicked document, l[hi] is the docid
            # for each clicked document, get documents above it
            doc_range = range(hi)
            for lo in doc_range:
                if lo in click_ids:  # y was clicked as well, no constraint.
                    continue
                # check current scores and update ranker weights as needed
                feature_diff = (self.current_query.get_feature_vector(
                    self.current_l[hi]) -
                    self.current_query.get_feature_vector(self.current_l[lo]))
                # hi is the document that should be ranked higher, so it should
                # have a higher score
                y = 1.0
                if y * dot(feature_diff, self.ranker.w.transpose()) < 1.0:
                    new_weights = (self.ranker.w + self.eta * y * feature_diff
                        - self.eta * self.lamb * self.ranker.w)
                    self.ranker.update_weights(new_weights)
        return self.ranker.w

    def get_solution(self):
        return self.ranker.w
