from numpy import *
import argparse
import logging
from random import gauss
import yaml
import glob
import os

from AbstractLearningSystem import AbstractLearningSystem
from ..utils import get_class


class SamplerSystem(AbstractLearningSystem):
    def __init__(self, feature_count, arg_str, run_count=""):
        logging.info("Initializing SamplerSystem")
        self.feature_count = feature_count
        # parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("-w", "--init_weights", help="Initialization "
            "colon seperated list of inital weight vectors, weight vectors are"
            " comma seperated", required=True)
        parser.add_argument("--sample_weights", default="sample_unit_sphere")
        parser.add_argument("--nr_rankers", type=int)
        parser.add_argument("-c", "--comparison", required=True)
        parser.add_argument("-f", "--comparison_args", nargs="*")
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
                logging.info("Loaded weight from file %s." % f)
        logging.info("Loaded %d weights." % len(weights))
        
        self.ranker_class = get_class(args["ranker"])
        if "ranker_args" in args and args["ranker_args"] != None:
            self.ranker_args = " ".join(args["ranker_args"])
            self.ranker_args = self.ranker_args.strip("\"")
        else:
            self.ranker_args = None
        self.ranker_tie = args["ranker_tie"]
        self.sample_weights = args["sample_weights"]
        self.rankers = [self.ranker_class(self.ranker_args,
                                          self.ranker_tie,
                                          feature_count,
                                          init=",".join([str(n) for n in w]),
                                          sample=self.sample_weights)
                        for w in weights]
        
        self.comparison_class = get_class(args["comparison"])
        if "comparison_args" in args and args["comparison_args"] != None:
            self.comparison_args = " ".join(args["comparison_args"])
            self.comparison_args = self.comparison_args.strip("\"")
        else:
            self.comparison_args = None
        self.comparison = self.comparison_class(self.comparison_args)
        
        self.r1 = 0     # One ranker to be compared in live evaluation.
        self.r2 = 0     # The other ranker to be compared against self.r1.
        
        sampler_class = get_class(args["sampler"])
        try:
            self.sampler = sampler_class(self.rankers, arg_str, run_count)
        except TypeError:
            self.sampler = sampler_class(self.rankers, arg_str)
        self.logging_frequency = 1000
        self.iteration = 1

    def get_ranked_list(self, query):
        self.r1, self.r2, i1, i2 = self.sampler.get_arms()
        i1s = [i1]
        i2s = [i2]
        #while self.r1 == self.r2:
            #self.iteration += 1
            #self.sampler.update_scores(self.r1, self.r2)
            #self.r1, self.r2, i1, i2 = self.sampler.get_arms()
            #i1s.append(i1)
            #i2s.append(i2)

        (l, context) = self.comparison.interleave(self.r1, self.r2,
                                                  query,
                                                  10)
        self.current_l = l
        self.current_context = context
        self.current_query = query
        return l, i1s, i2s

    def update_solution(self, clicks):
        outcome = self.comparison.infer_outcome(
                                                self.current_l,
                                                self.current_context,
                                                clicks,
                                                self.current_query)
        if outcome < 0:
            win = self.sampler.update_scores(self.r1, self.r2)#, -outcome)
        elif outcome > 0:
            win = self.sampler.update_scores(self.r2, self.r1)#, outcome)
        else:
            if gauss(0,1) > 0:
                win = self.sampler.update_scores(self.r1, self.r2)#, -outcome)
            else:
                win = self.sampler.update_scores(self.r2, self.r1)#, outcome)
        self.iteration += 1
        return win

    def get_solution(self):
        if self.iteration % self.logging_frequency == 0:
            logging.info("Iteration %d" % self.iteration)
        return self.sampler.get_winner().w
