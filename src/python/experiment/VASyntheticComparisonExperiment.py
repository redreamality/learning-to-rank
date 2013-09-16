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

import argparse
import yaml
from numpy import mean, where
import numpy
from random import randint, sample

from utils import get_class
from ranker import (SyntheticProbabilisticRankingFunction,
                    SyntheticDeterministicRankingFunction)
from document import Document


class VASyntheticComparisonExperiment():
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
        self.pareto_um_class = get_class("environment.FederatedClickModel")
        self.pareto_um = self.pareto_um_class(None)
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
                        (SyntheticDeterministicRankingFunction(ranker_args, # A
                        self.ties), SyntheticDeterministicRankingFunction(  # B
                        ranker_args, self.ties))
                elif ranker == "prob":
                    self.rankers[ranker][ranker_args] = \
                        (SyntheticProbabilisticRankingFunction(ranker_args, # A
                        self.ties), SyntheticProbabilisticRankingFunction(  # B
                        ranker_args, self.ties))
                else:
                    raise ValueError("Unknown ranker: " + ranker)
        # generate synthetic better and worse rankers
        (self.docids, self.labels) = self._generate_synthetic_documents(
            self.length, self.num_relevant)
        (self.better, self.worse, self.labels) = self._generate_synthetic_rankings_randomly(
            self.docids, self.labels, self.length,
            posmethod=args["vertical_posmethod"],
            docmethod=args["vertical_docmethod"],
            vertrel=args["vertical_vertrel"],
            blocksize=args["vertical_blocksize"],
            independentplacement=args["vertical_independentplacement"])

    def run(self):
        """Run the experiment for num_queries queries."""
        # initialize counts and outcome arrays
        outcomes = {}
        click_counts = {}
        block_counts = {}
        for method_id in self.methods:
            outcomes[method_id] = []
            click_counts[method_id] = []
            block_counts[method_id] = []
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
                l = l[:self.length]
                block_counts[method_id].append(self.block_count(l))
                clicks = self.um.get_clicks(l, self.labels)
                # init ranking again for comparisons
                better_ranker.init_ranking(list(self.better))
                worse_ranker.init_ranking(list(self.worse))
                l = [x[0] for x in l]
                o = method["instance"].infer_outcome(l, a, clicks, None)
                # record outcomes and number of clicks
                outcomes[method_id].append(float(o))
                click_counts[method_id].append(clicks.tolist().count(1))
        # record ranker pairs, comparison outcomes
        yaml.dump({
            "outcomes": outcomes,
            "click_counts": click_counts,
            "block_counts": block_counts
            }, self.log_fh, default_flow_style=False)

        # diagnose errors
        for method_id, method in self.methods.items():
            o = mean(outcomes[method_id])
            if o <= 0:
                print "\nUnexpected outcome:", o
                print method

    @staticmethod
    def block_count(l):
        inBlock = False
        count = 0
        for d in l:
            if d[1]:
                if not inBlock:
                    count += 1
                inBlock = True
            else:
                inBlock = False
        return count

    @staticmethod
    def _vertpos(r1, r2, posmethod, blocksize):
        if posmethod == "beyondten":
            # This distribution is taken from figure 2 in "Beyond Ten Blue
            # Links: Enabling User Click Modeling in Federated Web Search" by
            # Chen et al. The distribution is scaled back to the 1-10 interval.
            posdist = [0.13950538998097659, 0.027266962587190878,
                       0.24096385542168688, 0.1407736207989854,
                       0.030437539632213073, 0.019023462270133174,
                       0.0025364616360177474, 0.04438807863031058,
                       0.3119847812301839, 0.04311984781230175]
            pos = int(where(numpy.random.multinomial(1, posdist) == 1)[0])
            if pos + blocksize > min(len(r1), len(r2)):
                pos = min(len(r1), len(r2)) - blocksize
        elif posmethod == "uniform":
            pos = randint(0, min(len(r1), len(r2)) - blocksize)
        return pos

    @staticmethod
    def _set_vertical(r1, r2, olabels,
                      length,
                      posmethod="beyondten",
                      docmethod="insert",
                      vertrel="nonrel",
                      blocksize=3,
                      independentplacement=True):

        if independentplacement:
            pos1 = VASyntheticComparisonExperiment._vertpos(r1,
                                                            r2,
                                                            posmethod,
                                                            blocksize)
            pos2 = VASyntheticComparisonExperiment._vertpos(r1,
                                                            r2,
                                                            posmethod,
                                                            blocksize)
        else:
            pos1 = pos2 = VASyntheticComparisonExperiment._vertpos(r1,
                                                               r2,
                                                               posmethod,
                                                               blocksize)

        if docmethod == "assign":
            r1 = [d.set_type(pos1 <= i < (pos1 + blocksize))
                  for i, d in enumerate(r1)]
            r2 = [d.set_type(pos2 <= i < (pos2 + blocksize))
                  for i, d in enumerate(r2)]
        elif docmethod == "insert":
            maxid = max(r1 + r2)
            r1 = [d.set_type(False) for d in r1]
            r2 = [d.set_type(False) for d in r2]
            for i in range(blocksize):
                r1.insert(pos1, Document(maxid + i + 1, True))
                r2.insert(pos2, Document(maxid + i + 1, True))

        labels = olabels[:]
        for doc in set(r1 + r2):
            if not doc.get_type:
                continue

            vdoc = doc.get_id()
            if vdoc >= len(labels):
                labels += [0] * (vdoc - len(labels) + 1)

            if vertrel == "nonrel":
                labels[vdoc] = 0
            elif vertrel == "rel":
                labels[vdoc] = 1
            elif vertrel == "ratio":
                ratio = float(sum(olabels)) / length
                labels[vdoc] = numpy.random.binomial(1, ratio)

        return r1, r2, labels

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

        docids = [Document(x) for x in range(length)]
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

    def _pareto_dominates(self, a, b, labels):
        # Cut and sort by exmaniation probability accoridng to user model
        a = a[:self.length]
        b = b[:self.length]
        examination_a = self.pareto_um.get_examination_prob(a)
        examination_b = self.pareto_um.get_examination_prob(b)
        a.sort(key=dict(zip(a, examination_a)).get, reverse=True)
        b.sort(key=dict(zip(b, examination_b)).get, reverse=True)

        rel_a = [index for index, item in enumerate(a) if labels[item.get_id()] == 1]
        rel_b = [index for index, item in enumerate(b) if labels[item.get_id()] == 1]
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

    def _generate_synthetic_rankings_randomly(self,
                                              docids, olabels,
                                              length,
                                              posmethod,
                                              docmethod,
                                              vertrel,
                                              blocksize,
                                              independentplacement):
        """Generate synthetic documents rankings that implement pareto
        dominance. there needs to be at least one non-relevant document,
        otherwise no better / worse ranking pair can be constructed.
        Returns (better_ranking, worse_ranking)."""

        assert(len(docids) > 0)
        assert(len(docids) == len(olabels))
        assert(0 in olabels)
        assert(1 in olabels)

        for _ in range(1000):
            labels = olabels[:]
            a = VASyntheticComparisonExperiment._random_permutation(docids)
            b = VASyntheticComparisonExperiment._random_permutation(docids)

            a, b, labels = VASyntheticComparisonExperiment._set_vertical(
                                    a, b, labels, length,
                                    posmethod=posmethod,
                                    docmethod=docmethod,
                                    vertrel=vertrel,
                                    blocksize=blocksize,
                                    independentplacement=independentplacement)

            if self._pareto_dominates(a, b, labels):
                return (a, b, labels)
            elif self._pareto_dominates(b, a, labels):
                return (b, a, labels)
        raise(ValueError, "Could not find pareto dominated ranker for labels "
              "%s after 1000 trials." % ", ".join([str(x) for x in labels]))
