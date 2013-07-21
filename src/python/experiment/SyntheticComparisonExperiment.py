# KH, 2013/04/02
# Runs a comparison experiment on synthetic data

import argparse
import yaml

from numpy import array, cumsum, delete, mean
from random import randint, random, sample
from utils import get_class


class SyntheticComparisonExperiment():
    """Represents an experiment in which synthetic rankers are compared to
    investigate theoretical properties / guarantees.
    """

    def __init__(self, log_fh, args):
        """Initialize an experiment using the provided arguments."""
        self.log_fh = log_fh
        self.ties = args["ties"] if "ties" in args else "first"
        # additional configuration: number of relevant documents
        # (number or "random")
        self.length = args["result_length"]
        self.num_relevant = args["num_relevant"]
        self.num_queries = args["num_queries"]
        self.um_class = get_class(args["user_model"])
        self.um_args = args["user_model_args"]
        self.um = self.um_class(self.um_args)
        # initialize interleaved comparison methods according to configuration
        parser = argparse.ArgumentParser(description="parse arguments of an "
            "evaluation method.", prog="evaluation method configuration")
        parser.add_argument("-c", "--class_name")
        parser.add_argument("-r", "--ranker", help="can be 'det' or 'prob'")
        parser.add_argument("-a", "--ranker_args")
        parser.add_argument("-i", "--interleave_method")
        self.rankers = {}
        self.methods = {}
        # init live methods
        if "evaluation_methods" in args:
            for method_id, method in enumerate(
                    args["evaluation_methods"]):
                self.methods[method] = {}
                method_args_str = \
                    args["evaluation_methods_args"][method_id]
                method_args = vars(parser.parse_known_args(
                    method_args_str.split())[0])
                class_name = method_args["class_name"]
                self.methods[method]["instance"] = \
                    get_class(class_name)(method_args_str)
                ranker = method_args["ranker"]
                ranker_args = method_args["ranker_args"]
                self.methods[method]["ranker"] = ranker
                self.methods[method]["ranker_args"] = ranker_args
                if not ranker in self.rankers:
                    self.rankers[ranker] = {}
                if not ranker_args in self.rankers[ranker]:
                    self.rankers[ranker][ranker_args] = {}
        # init rankers needed by the comparison methods. rankers can be
        # deterministic (det) or probabilistic (prob), and can have different
        # arguments
        for ranker in self.rankers:
            for ranker_args in self.rankers[ranker]:
                if ranker == "det":
                    self.rankers[ranker][ranker_args] = \
                        (self.SyntheticDeterministicRanker(ranker_args,  # A
                        self.ties), self.SyntheticDeterministicRanker(   # B
                        ranker_args, self.ties))
                elif ranker == "prob":
                    self.rankers[ranker][ranker_args] = \
                        (self.SyntheticProbabilisticRanker(ranker_args,  # A
                        self.ties), self.SyntheticProbabilisticRanker(   # B
                        ranker_args, self.ties))
                else:
                    raise ValueError("Unknown ranker: " + ranker)
        # generate synthetic better and worse rankers
        (self.docids, self.labels) = self._generate_synthetic_documents(
            self.length, self.num_relevant)
        (self.better, self.worse) = self._generate_synthetic_rankings_randomly(
            self.docids, self.labels)
        print self.labels
        print self.better
        print self.worse

    def run(self):
        """Run the experiment for num_queries queries."""
        # initialize counts and outcome arrays
        outcomes = {}
        click_counts = {}
        for method_id in self.methods:
            outcomes[method_id] = []
            click_counts[method_id] = []
        # compare better and worse ranker on num_queries impressions
        for _ in range(self.num_queries):
            for method_id, method in self.methods.items():
                (better_ranker, worse_ranker) = self.rankers[method["ranker"]][
                    method["ranker_args"]]
                better_ranker.init_ranking(list(self.better))
                worse_ranker.init_ranking(list(self.worse))
                # interleave known worse and better rankers (outcomes should
                # converge to 1)
                (l, a) = method["instance"].interleave(worse_ranker,
                    better_ranker, None, self.length)
                clicks = self.um.get_clicks(l, self.labels)
                # init ranking again for comparisons
                better_ranker.init_ranking(list(self.better))
                worse_ranker.init_ranking(list(self.worse))
                o = method["instance"].infer_outcome(l, a, clicks, None)
                # record outcomes and number of clicks
                outcomes[method_id].append(float(o))
                click_counts[method_id].append(clicks.tolist().count(1))

        # record ranker pairs, comparison outcomes
        yaml.dump({
            "outcomes": outcomes,
            "click_counts": click_counts
            }, self.log_fh, default_flow_style=False)

        # diagnose errors
        for method_id, method in self.methods.items():
            o = mean(outcomes[method_id])
            if o <= 0:
                print "\nUnexpected outcome:", o
                print method

    @staticmethod
    def _generate_synthetic_documents(length, num_relevant):
        """Generate a synthetic document list of <length> with <num_relevant>
        relevant documents."""

        if num_relevant == "random":
            num_relevant = randint(1, length / 2)
        elif "-" in num_relevant:
            min_rel, max_rel = num_relevant.split("-")
            num_relevant = randint(int(min_rel), int(max_rel))
        else:
            num_relevant = int(num_relevant)

        assert(length > 0)
        assert(num_relevant > 0)
        assert(num_relevant < length)

        docids = range(length)
        labels = [0] * length
        nonrel = set(docids)
        rel = set()

        while (len(docids) - len(nonrel)) < num_relevant:
            next_rel = sample(nonrel, 1)[0]
            labels[next_rel] = 1
            nonrel.remove(next_rel)
            rel.add(next_rel)

        return (docids, labels)

    @staticmethod
    def _random_permutation(iterable, r=None):
        """Random selection from itertools.permutations(iterable, r).
        From: http://docs.python.org/2/library/itertools.html"""
        pool = tuple(iterable)
        r = len(pool) if r is None else r
        return tuple(sample(pool, r))

    @staticmethod
    def _pareto_dominates(a, b, labels):
        rel_a = [index for index, item in enumerate(a) if labels[item] == 1]
        rel_b = [index for index, item in enumerate(b) if labels[item] == 1]
        # if a has fewer relevant documents it cannot dominate b
        if len(rel_a) < len(rel_b):
            return False
        distance = 0
        for index_a, index_b in zip(rel_a, rel_b):
            if index_a > index_b:
                return False
            elif index_a < index_b:
                distance += index_b - index_a
            # if b has fewer relevant documents and none of its elements
            # violate pareto dominance
            if len(rel_a) > len(rel_b) and index_b == rel_b[-1]:
                return True

        if distance > 0:
            return True
        return False

    @staticmethod
    def _generate_synthetic_rankings_randomly(docids, labels):
        """Generate synthetic documents rankings that implement pareto
        dominance. there needs to be at least one non-relevant document,
        otherwise no better / worse ranking pair can be constructed.
        Returns (better_ranking, worse_ranking)."""

        assert(len(docids) > 0)
        assert(len(docids) == len(labels))
        assert(0 in labels)
        assert(1 in labels)

        for _ in range(1000):
            a = SyntheticComparisonExperiment._random_permutation(docids)
            b = SyntheticComparisonExperiment._random_permutation(docids)

            if SyntheticComparisonExperiment._pareto_dominates(a, b, labels):
                return (list(a), list(b))
            elif SyntheticComparisonExperiment._pareto_dominates(b, a, labels):
                return (list(b), list(a))
        raise(ValueError, "Could not find pareto dominated ranker for labels "
              "%s after 1000 trials." % ", ".join([str(x) for x in labels]))

    class SyntheticProbabilisticRanker:
        """Synthetic ranker for use in this experiment only"""

        def __init__(self, ranker_arg_str, ties="random"):
            self.ranker_type = float(ranker_arg_str)
            self.ties = ties

        def init_ranking(self, synthetic_docids):
            if not synthetic_docids:
                return
            # assume that synthetic_docids are in rank order
            self.docids = synthetic_docids
            ranks = array(range(1, len(self.docids) + 1))
            # determine probabilities based on (reverse) document ranks
            tmp_val = 1. / pow(ranks, self.ranker_type)
            self.probs = tmp_val / sum(tmp_val)

        def document_count(self):
            return len(self.docids)

        def next(self):
            """produce the next document by random sampling, or
            deterministically"""
            # if there are no more documents
            if len(self.docids) < 1:
                raise Exception("There are no more documents to be selected")

            # if there's only one document
            if len(self.docids) == 1:
                self.probs = delete(self.probs, 0)  # should be empty now
                pick = self.docids.pop()  # pop, because it's a list
                return pick

            # sample if there are more documents
            cumprobs = cumsum(self.probs)
            pick = -1
            rand = random()  # produces a float in range [0.0, 1.0)
            for pos, cp in enumerate(cumprobs):
                if rand < cp:
                    pick = self.docids.pop(pos)  # pop, because it's a list
                    break

            if (pick == -1):
                print "Cumprobs:", cumprobs
                print "rand", rand
                raise Exception("Could not select document!")
            # renormalize
            self.probs = delete(self.probs, pos)  # delete, it's a numpy array
            self.probs = self.probs / sum(self.probs)
            return pick

        def next_det(self):
            pos = 0  # first is the most likely document
            pick = self.docids.pop(pos)
            # renormalize
            self.probs = delete(self.probs, pos)  # delete, it's a numpy array
            self.probs = self.probs / sum(self.probs)
            return pick

        def next_random(self):
            """produce a random next document"""

            # if there are no more documents
            if len(self.docids) < 1:
                raise Exception("There are no more documents to be selected")
            # otherwise, return a random document
            rn = randint(0, len(self.docids) - 1)
            return self.docids.pop(rn)

        def get_document_probability(self, docid):
            """get probability of producing doc as the next document drawn"""
            pos = self.docids.index(docid)
            return self.probs[pos]

        def rm_document(self, docid):
            """remove doc from list of available docs, adjust probabilities"""
            # find position of the document
            try:
                pos = self.docids.index(docid)
            except ValueError:
                print "cannot remove", docid,
                print "current document list:", self.docids
                print "qid:", self.qid
            # delete doc and renormalize
            self.docids.pop(pos)
            self.probs = delete(self.probs, pos)
            self.probs = self.probs / sum(self.probs)

        def update_weights(self, new_weights):
            """not required under synthetic data"""
            pass

    class SyntheticDeterministicRanker:
        """"synthetic deterministic ranker for use in this experiment"""
        def __init__(self, ranker_arg_str, ties="random"):
            self.ties = ties

        def init_ranking(self, synthetic_docids):
            if not synthetic_docids:
                return
            self.docids = synthetic_docids

        def document_count(self):
            return len(self.docids)

        def next(self):
            """produce the next document"""

            # if there are no more documents
            if len(self.docids) < 1:
                raise Exception("There are no more documents to be selected")
            # otherwise, return highest ranked document
            return self.docids.pop(0)  # pop first element

        def next_det(self):
            return self.next()

        def next_random(self):
            """produce a random next document"""

            # if there are no more documents
            if len(self.docids) < 1:
                raise Exception("There are no more documents to be selected")
            # otherwise, return a random document
            rn = randint(0, len(self.docids) - 1)
            return self.docids.pop(rn)

        def get_document_probability(self, docid):
            """get probability of producing doc as the next document drawn"""
            pos = self.docids.index(docid)
            return 1.0 if pos == 0 else 0.0

        def rm_document(self, docid):
            """remove doc from list of available docs, adjust probabilities"""
            # find position of the document
            pos = self.docids.index(docid)
            # delete doc and renormalize
            self.docids.pop(pos)

        def update_weights(self, new_weights):
            """not required under synthetic data"""
            pass
