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

from numpy import *
import argparse
import logging
from random import gauss
import yaml
import glob
import os

from AbstractLearningSystem import AbstractLearningSystem
from utils import get_class


class BaselineSamplerSystem(AbstractLearningSystem):
    def __init__(self, feature_count, arg_str):
        logging.info("Initializing SamplerSystem")
        self.feature_count = feature_count
        # parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("-w", "--init_weights", help="Initialization "
            "colon seperated list of inital weight vectors, weight vectors are"
            " comma seperated", required=True)
        parser.add_argument("--nr_rankers", type=int)
        parser.add_argument("--nr_results", type=int, default=10)
        parser.add_argument("-c", "--comparison", required=True)
        parser.add_argument("-f", "--comparison_args", nargs="*",
                            action="append")
        parser.add_argument("-r", "--ranker", required=True)
        parser.add_argument("-s", "--ranker_args", nargs="*")
        parser.add_argument("-t", "--ranker_tie", default="random")
        parser.add_argument("--sampler", required=True)

        args = vars(parser.parse_known_args(arg_str.split())[0])
        # initialize weights, comparison method, and learner

        weights = []
        if args["init_weights"].startswith("random"):
            for i in range(args["nr_rankers"]):
                v = zeros(self.feature_count)
                for i in range(0, self.feature_count):
                    v[i] = gauss(0, 1)
                weights.append(list(v))
        if args["init_weights"].startswith("selected"):
            selection = [4, 24, 39, 41, 50]
            for i in range(args["nr_rankers"]):
                v = zeros(self.feature_count)
                v[selection[i]] = 1.0
                weights.append(list(v))
        else:
            for f in sorted(glob.glob(os.path.join(args["init_weights"],
                                            "*.txt")))[:args["nr_rankers"]]:
                yamldata = yaml.load(open(f, 'r'))
                weight = yamldata["final_weights"]
                if len(weight) != feature_count:
                    raise Exception("List of initial weights does not have the"
                        " expected length (length is %d, expected %d)." %
                        (len(weight), feature_count))
                weights.append(weight)
        logging.info("Loaded %d weights." % len(weights))
        
        self.ranker_class = get_class(args["ranker"])
        if "ranker_args" in args and args["ranker_args"] != None:
            self.ranker_args = " ".join(args["ranker_args"])
            self.ranker_args = self.ranker_args.strip("\"")
        else:
            self.ranker_args = None
        self.ranker_tie = args["ranker_tie"]
        self.rankers = [self.ranker_class(self.ranker_args,
                                          self.ranker_tie,
                                          self.feature_count,
                                          init=",".join([str(x) for x in w]))
                        for w in weights]
        
        self.comparison_class = get_class(args["comparison"])
        if "comparison_args" in args and args["comparison_args"] != None:
            self.comparison_args = " ".join([item for sub in
                                             args["comparison_args"]
                                             for item in sub])
            self.comparison_args = self.comparison_args.replace("\"", "")
        else:
            self.comparison_args = None
        self.comparison = self.comparison_class(self.comparison_args)
        
        self.r1 = 0     # One ranker to be compared in live evaluation.
        self.r2 = 0     # The other ranker to be compared against self.r1.
        
        sampler_class = get_class(args["sampler"])
        self.sampler = sampler_class(self.rankers, arg_str)
        
        self.nr_results = args["nr_results"]
        self.iteration = 0

    def get_ranked_list(self, query):
        self.r1, self.r2, _, _ = self.sampler.get_arms()

        (l, context) = self.comparison.interleave(self.r1, self.r2,
                                                  query,
                                                  self.nr_results)
        self.current_l = l
        self.current_context = context
        self.current_query = query
        return l

    def update_solution(self, clicks):
        outcome = self.comparison.infer_outcome(self.current_l,
                                                self.current_context,
                                                clicks,
                                                self.current_query)
        if outcome <= 0:
            self.sampler.update_scores(self.r1, self.r2, score=1, play=1)
        else:
            self.sampler.update_scores(self.r2, self.r1, score=1, play=1)

        self.iteration += 1
        return self.get_solution()

    def get_solution(self):
        #logging.info("Iteration %d" % self.iteration)
        return self.sampler.get_sheet()
