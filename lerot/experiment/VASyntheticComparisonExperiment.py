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
from collections import defaultdict
import copy
import glob
import itertools
import json
import numpy as np
import os
import random
import scipy
import scipy.stats
import sys
import traceback

from ..document import Document
from ..ranker import ModelRankingFunction, SyntheticDeterministicRankingFunction, StatelessRankingFunction
from ..utils import get_class


VERTICALS = ["Web", "News", "Image", "Video"]

T_TEST_THRESHOLD_ALPHA = 0.05

# Number of times we run a click model to compute online metrics.
CLICK_METRIC_RUNS = 50


def normalize(l):
    s = sum(l)
    for i in range(len(l)):
        l[i] /= s

class VASyntheticComparisonExperiment():
    """Represents an experiment in which synthetic rankers are compared to
    investigate theoretical properties / guarantees.
    """

    def __init__(self, log_fh, args):
        """Initialize an experiment using the provided arguments."""
        self.verbose = args["verbose"]
        self.log_fh = log_fh
        # additional configuration: number of relevant documents
        # (number or "random")
        self.length = args["result_length"]
        self.rankings = args["rankings"]
        self.block_size = args["vertical_blocksize"]
        self.um_class = get_class(args["user_model"])
        self.um_args = args["user_model_args"]
        self.um = self.um_class(self.um_args)
        self.pareto_um_class = get_class(args["pareto_um_class"])
        self.pareto_um = self.pareto_um_class(args["pareto_um_args"])
        self.fixed_doc_placement = (args["vertical_placement"]=="fixed")
        # initialize interleaved comparison methods according to configuration
        parser = argparse.ArgumentParser(
            description="parse arguments of an evaluation method.",
            prog="evaluation method configuration")
        parser.add_argument("--class_name")
        parser.add_argument("-i", "--interleave_method")
        self.methods = {}
        # init live methods
        if "evaluation_methods" in args:
            for method_id, method in enumerate(
                    args["evaluation_methods"]):
                self.methods[method] = {}
                method_args_str = \
                    args["evaluation_methods_args"][method_id]
                method_args = vars(parser.parse_known_args(method_args_str.split())[0])
                class_name = method_args["class_name"]
                self.methods[method]["instance"] = \
                    get_class(class_name)(method_args_str)
        self.offline_metrics = {}
        for metric in args.get("offline_metrics", "").split(","):
            metric = metric.rstrip()
            if metric:
                metric_short = metric.split(".")[-1].split("Eval")[0]
                self.offline_metrics[metric_short] = get_class(metric)()

        self.online_metrics = set([])    # will be filled later
        self.compute_online_metrics = args["compute_online_metrics"]

        self.compute_interleaved_metrics = args["compute_interleaved_metrics"]

        self.num_repeat_interleaving = args["num_repeat_interleaving"]
        self.random_query_draw = args.get("num_random_draws") is not None

        self.system_comparison = args["system_comparison"]
        if self.system_comparison == "none":
            dominates = lambda a, b, l: True
        elif self.system_comparison == "pareto":
            dominates = self._pareto_dominates
        else:
            raise Exception(
                "Unsupported system comparison: %s" % self.system_comparison
            )
        if self.rankings == "synthetic":
            documents_A, documents_B, self.labels = (
                VASyntheticComparisonExperiment.generate_ranking_pair(
                    self.length, args["num_relevant"],
                    pos_method=args["vertical_posmethod"],
                    vert_rel=args["vertical_vertrel"],
                    block_size=self.block_size,
                    verticals=[v.strip() for v in args["verticals"].split(",")],
                    fixed=self.fixed_doc_placement,
                    dominates=dominates
                )
            )
            self.rankers = (
                SyntheticDeterministicRankingFunction(documents_A),
                SyntheticDeterministicRankingFunction(documents_B)
            )

            self.topics = ["dummy"]
        elif self.rankings == "model":
            # TODO(chuklin): make self.qrels, vert_map, self.Iprob to be
            # lazily initialized class attributes.

            self.Iprob = {}
            if args.get("vert_map_file") is not None and args.get("Iprob_file") is not None:
                vert_map = {}
                with open(args["vert_map_file"]) as vert_map_f:
                    for line in vert_map_f:
                        vertical, vert_id = line.split()
                        vert_map[vert_id] = vertical
                with open(args["Iprob_file"]) as Iprob:
                    for line in Iprob:
                        topic, vert_id, prob = line.split()
                        self.Iprob.setdefault(topic, {})
                        self.Iprob[topic][vert_map[vert_id]] = float(prob)
            else:
                self.Iprob = defaultdict(lambda: defaultdict(lambda: 1.0))

            page_files = glob.glob(os.path.join(os.path.abspath(args["run_dir"]), "*"))
            page_files.sort()
            total_rankers = len(page_files)
            # indexes of rankers to compare
            all_pairs = []
            for i in xrange(1, total_rankers):
                for j in xrange(i):
                    all_pairs.append((i, j))
            i, j = all_pairs[args["ranker_pair_idx"]]
            print "Comparing rankers %s and %s" % (page_files[i], page_files[j])
            self.rankers = (ModelRankingFunction(), ModelRankingFunction())
            self.doc_to_id = {}
            self.current_doc_id = 0
            self.topics = set([])
            self.ideal_ranker = ModelRankingFunction()
            # Add ideal page to the list of ranker files such that it can be
            # accessed using -1 index.
            page_files.append(args.get("ideal_page_as_rbp"))
            for r, idx in zip(
                    [self.rankers[0], self.rankers[1], self.ideal_ranker],
                    [i, j, -1]):
                if page_files[idx] is None:
                    continue
                with open(page_files[idx]) as f:
                    current_file_query_docs = set([])
                    for line in f:
                        topic, _, doc, rank, system, vertical = line.split()
                        self.topics.add(topic)
                        if (topic, doc) in current_file_query_docs:
                            # Skip duplicate docs
                            continue
                        current_file_query_docs.add((topic, doc))
                        if doc not in self.doc_to_id:
                            self.doc_to_id[doc] = self.current_doc_id
                            self.current_doc_id += 1
                        doc_id = self.doc_to_id[doc]
                        r.add_doc_for_query(topic, Document(doc_id, vertical))
            self.qrels = {}
            if args.get("qrel_file") is not None:
                with open(args["qrel_file"]) as qrels:
                    for line in qrels:
                        topic, doc, rel = line.split()
                        self.qrels.setdefault(topic, defaultdict(lambda: 0))
                        if doc in self.doc_to_id:
                            self.qrels[topic][self.doc_to_id[doc]] = int(rel[1:])
            else:
                self.qrels = defaultdict(lambda: defaultdict(lambda: 0.0))
            if "ideal_page_as_rbp" in args and "AsRbp" in self.offline_metrics:
                self.ideal_as_rbp = dict((q, self.offline_metrics["AsRbp"].get_value(
                        ranking, self.qrels[q], self.Iprob[q], self.length)) \
                                for (q, ranking) in self.ideal_ranker.pages.iteritems()
                )
            self.topics = sorted(self.topics)
        else:
            raise Exception("unknown input rankings specified: %s" % args["rankings"])

    def init_rankers(self, query):
        """ Init rankers for a query

            Since the ranker may be stateful, we need to init it every time we
            access its documents.
        """
        self.rankers[0].init_ranking(query)
        self.rankers[1].init_ranking(query)


    def run(self):
        output = {
            "outcomes":     defaultdict(lambda: []),
            "relaxed":      defaultdict(lambda: []),
            "click_counts": defaultdict(lambda: []),
            "block_counts": defaultdict(lambda: []),
            "block_pos":    defaultdict(lambda: []),
            "block_sizes":  defaultdict(lambda: []),
        }
        result_length = self.length
        if self.fixed_doc_placement and self.rankings == "synthetic":
            result_length += self.block_size
        for r in self.rankers:
            assert isinstance(r, StatelessRankingFunction)

        topics = [random.choice(self.topics)] if self.random_query_draw else self.topics

        for topic in topics:
            if self.rankings == "synthetic":
                labels = self.labels
                orientations = defaultdict(lambda: 1.0)
            else: # model
                labels = self.qrels[topic]
                orientations = self.Iprob[topic]

            as_rbp_denominator = 1.0
            if self.rankings == "model" and "AsRbp" in self.offline_metrics:
                as_rbp_denominator = self.ideal_as_rbp[topic]
                if as_rbp_denominator == 0.0:
                    as_rbp_denominator = 1.0

            for metric_name, metric_instance in self.offline_metrics.iteritems():
                output.setdefault(metric_name, defaultdict(lambda: []))
                self.init_rankers(topic)
                if metric_name == "RP" and self.rankings == "model":
                    metric_A, metric_B = [metric_instance.get_value(
                            self.rankers[i].getDocs(result_length),
                            labels, orientations, result_length,
                            ideal_ranking=self.ideal_ranker.pages[topic])
                        for i in [0, 1]
                    ]
                else:
                    metric_A = metric_instance.get_value(
                            self.rankers[0].getDocs(result_length),
                            labels, orientations, result_length)
                    metric_B = metric_instance.get_value(
                            self.rankers[1].getDocs(result_length),
                            labels, orientations, result_length)
                    if metric_name == "AsRbp" and self.rankings == "model":
                        metric_A /= as_rbp_denominator
                        metric_B /= as_rbp_denominator
                # Repeat the value n times such that metrics of the original
                # and the interleaved systems have the same lengths.
                output[metric_name]["A"].append(metric_A)
                output[metric_name]["B"].append(metric_B)
            if self.system_comparison == "pareto":
                # Store pareto-dominance similar to how we store metric values
                # so that they can be accessed uniformly.
                output.setdefault("pareto", defaultdict(lambda: []))
                self.init_rankers(topic)
                docsA = self.rankers[0].getDocs(result_length)
                docsB = self.rankers[1].getDocs(result_length)
                if self._pareto_dominates(docsA, docsB, labels,
                        orientation=orientations):
                    output["pareto"]["A"].append(1)
                    output["pareto"]["B"].append(0)
                elif self._pareto_dominates(docsB, docsA, labels,
                        orientation=orientations):
                    output["pareto"]["A"].append(0)
                    output["pareto"]["B"].append(1)
                else:
                    output["pareto"]["A"].append(0)
                    output["pareto"]["B"].append(0)
                    continue
            if self.compute_online_metrics:
                self.init_rankers(topic)
                for side, ranker in zip(["A", "B"], self.rankers):
                    docs = ranker.getDocs(result_length)
                    for run in xrange(CLICK_METRIC_RUNS):
                        online_metrics = self.get_online_metrics(
                            self.um.get_clicks(docs, labels, orientation=orientations),
                            docs
                        )
                        for metric_name, v in online_metrics.iteritems():
                            output.setdefault(metric_name,
                                    defaultdict(lambda: []))
                            output[metric_name][side].append(v)
                            self.online_metrics.add(metric_name)

            for method_id, method in self.methods.iteritems():
                self.init_rankers(topic)
                current_verticals = (self.rankers[0].verticals(result_length) |
                                     self.rankers[1].verticals(result_length))
                try:
                    # Interleave should call reset for query
                    interleaver = method["instance"]
                    n_interleavings = interleaver.interleave_n(
                        self.rankers[1],
                        self.rankers[0],
                        topic,
                        result_length,
                        self.num_repeat_interleaving
                    )
                    if hasattr(interleaver, "relaxed"):
                        output["relaxed"][method_id].append(1 if interleaver.relaxed else 0)
                except Exception as e:
                    print >>sys.stderr, method_id, "failed. Exception =", e
                    self.init_rankers(topic)
                    print >>sys.stderr, ("Topic:", topic, "Rankings:",
                            self.rankers[0].getDocs(), self.rankers[1].getDocs())
                    traceback.print_exc()
                    continue
                for (interleaved, a) in n_interleavings:
                    # record outcomes and number of clicks
                    output["outcomes"][method_id].append(
                        method["instance"].infer_outcome(
                            [x.get_id() for x in interleaved],
                            a,
                            self.um.get_clicks(interleaved, labels, orientation=orientations),
                            None
                        )
                    )
                    if self.compute_interleaved_metrics:
                        block_cnts = VASyntheticComparisonExperiment.block_counts(interleaved)
                        block_szs = VASyntheticComparisonExperiment.block_sizes(interleaved)
                        for vert in current_verticals:
                            output["block_counts"][method_id].append(block_cnts[vert])
                            output["block_sizes"][method_id].append(
                                    float(block_szs[vert]) / self.block_size)
                        output["block_pos"][method_id].append(
                                VASyntheticComparisonExperiment.block_position1(interleaved, result_length))
                        for metric_name, metric_instance in self.offline_metrics.iteritems():
                            if metric_name == "RP" and self.rankings == "model":
                                metric_L = metric_instance.get_value(
                                    interleaved, labels, orientations, result_length,
                                    ideal_ranking=self.ideal_ranker.pages[topic])
                            else:
                                metric_L = metric_instance.get_value(
                                        interleaved, labels, orientations, result_length)
                                if metric_name == "AsRbp" and self.rankings == "model":
                                    metric_L /= as_rbp_denominator
                            output[metric_name][method_id].append(metric_L)
                        if self.compute_online_metrics:
                            for run in xrange(CLICK_METRIC_RUNS):
                                online_metrics = self.get_online_metrics(
                                    self.um.get_clicks(interleaved, labels, orientation=orientations),
                                    interleaved
                                )
                                for metric_name, v in online_metrics.iteritems():
                                    output[metric_name][method_id].append(v)

        if self.compute_interleaved_metrics:
            for metric_name in itertools.chain(
                    self.offline_metrics.iterkeys(), self.online_metrics):
                side_metrics = {}
                for side in ["A", "B"]:
                    side_metrics[side] = np.mean(output[metric_name][side])
                # A is always the side with higher metric value; swap if needed
                A, B = sorted(["A", "B"], key=lambda side: side_metrics[side],
                              reverse=True)
                side_metrics["A"] = A
                side_metrics["B"] = B
                for method_id in self.methods.iterkeys():
                    vals = output[metric_name][method_id]
                    mean_val = np.mean(vals)
                    degrades = False
                    if mean_val < side_metrics[B]:
                        _, p_value = scipy.stats.ttest_rel(vals, output[metric_name][B])
                        if p_value <= T_TEST_THRESHOLD_ALPHA:
                            degrades = True
                    output[metric_name][method_id + "_degrades"] = [1 if degrades else 0]

        for method, o in output["outcomes"].iteritems():
            N = len(o)
            for part in xrange(5):
                num_impr = N // 5 * (part + 1)
                name = "outcomes_significant_%d" % num_impr
                output.setdefault(name, defaultdict(lambda: []))
                output[name][method] = [1] if scipy.stats.ttest_1samp(o[:num_impr], 0.0)[1] <= T_TEST_THRESHOLD_ALPHA else [0]

        # record ranker pairs, comparison outcomes
        json.dump(output, self.log_fh)

    @staticmethod
    def block_counts(l):
        counts = defaultdict(lambda: 0)
        for i in xrange(len(l)):
            cur_type = l[i].get_type()
            if cur_type != "Web" and (
                    i + 1 == len(l) or cur_type != l[i + 1].get_type()):
                counts[l[i].get_type()] += 1
        return counts

    @staticmethod
    def block_position1(l, result_length):
        for i in xrange(len(l)):
            cur_type = l[i].get_type()
            if cur_type != "Web":
                return i
        return result_length

    @staticmethod
    def block_sizes(l):
        sizes = defaultdict(lambda: 0)
        for d in l:
            cur_type = d.get_type()
            if cur_type != "Web":
                sizes[cur_type] += 1
        return sizes

    @staticmethod
    def _vertpos(min_len, pos_method):
        if pos_method == "beyondten":
            # This distribution is taken from figure 2 in "Beyond Ten Blue
            # Links: Enabling User Click Modeling in Federated Web Search" by
            # Chen et al. The distribution is scaled back to the 1-10 interval.
            posdist = [0.13950538998097659, 0.027266962587190878,
                       0.24096385542168688, 0.1407736207989854,
                       0.030437539632213073, 0.019023462270133174,
                       0.0025364616360177474, 0.04438807863031058,
                       0.3119847812301839, 0.04311984781230175]
            pos = int(np.where(np.random.multinomial(1, posdist) == 1)[0])
        elif pos_method == "uniform":
            pos = random.randint(0, min_len)
        return pos

    @staticmethod
    def _set_vertical(r1, r2, olabels, length,
                      pos_method="beyondten",
                      vert_rel="non-relevant",
                      block_size=3,
                      verticals=None):
        if verticals is None:
            verticals = [VERTICALS[1]]
        # Select position to insert vertical blocks (at random).
        positions = {}
        for vert in verticals:
            pos = VASyntheticComparisonExperiment._vertpos(
                    min(len(r1), len(r2)), pos_method)
            positions[vert] = pos

        max_id = max(d.get_id() for d in r1 + r2)
        r1 = [d.set_type("Web") for d in r1]
        r2 = [d.set_type("Web") for d in r2]
        for vert in verticals:
            pos = positions[vert]
            for i in range(block_size):
                r1.insert(pos, Document(max_id + i + 1, vert))
                r2.insert(pos, Document(max_id + i + 1, vert))
            # Adjust positions according to the just inserted block.
            del positions[vert]
            max_id += block_size
            for v, p in positions.iteritems():
                if p >= pos:
                    positions[v] += block_size

        assert all(all(v == 1 for v in
                VASyntheticComparisonExperiment.block_counts(r).itervalues()) for r in [r1, r2])

        labels = olabels[:]
        for doc in set(r1 + r2):
            if doc.get_type() == "Web":
                continue
            vdoc = doc.get_id()
            if vdoc >= len(labels):
                labels += [0] * (vdoc - len(labels) + 1)

            if vert_rel == "non-relevant":
                labels[vdoc] = 0
            elif vert_rel == "all-relevant":
                labels[vdoc] = 1
            elif vert_rel == "relevant":
                ratio = float(sum(olabels)) / length
                labels[vdoc] = np.random.binomial(1, ratio)

        return r1, r2, labels

    @staticmethod
    def generate_ranking_pair(result_length, num_relevant,
                              pos_method="beyondten", vert_rel="non-relevant",
                              block_size=3, verticals=None,
                              fixed=False, dominates=lambda a, b, l: True):
        """ Generate pair of synthetic rankings.
            Appendix A, https://bitbucket.org/varepsilon/tois2013-interleaving
        """
        if verticals is None:
            verticals = [VERTICALS[1]]
        # This is what is called 'd' in the paper.
        pool_length = result_length + 2
        if fixed:
            document_pool, labels = (
                VASyntheticComparisonExperiment._generate_document_pool(
                        pool_length, num_relevant))
        else:
            assert block_size * len(verticals) <= pool_length
            prob = float(block_size) / pool_length
            document_pool, labels = (
                VASyntheticComparisonExperiment._generate_document_pool(
                        pool_length, num_relevant,
                        verticals, [prob] * len(verticals), vert_rel))
        for _ in xrange(1000):
            ranking_A = VASyntheticComparisonExperiment._generate_ranking(document_pool)
            ranking_B = VASyntheticComparisonExperiment._generate_ranking(document_pool)
            if fixed:
                ranking_A, ranking_B, labels = (
                        VASyntheticComparisonExperiment._set_vertical(
                                ranking_A, ranking_B, labels, result_length,
                                pos_method, vert_rel,
                                block_size, verticals))
            if dominates(ranking_A, ranking_B, labels):
                return ranking_A, ranking_B, labels
            elif dominates(ranking_B, ranking_A, labels):
                return ranking_B, ranking_A, labels
        raise Exception("Could not find pareto dominated ranker for labels "
              "%s after 1000 trials." % ", ".join(str(x) for x in labels))

    @staticmethod
    def _generate_document_pool(num_documents, num_relevant,
                                verticals=None, prob_of_doc_vert=None,
                                vert_rel="non-relevant"):
        """
            num_documents --- total number of documents in the pool.
                              The closer it is to the serp_size the more similar
                              two generated SERPs would be.
            num_relevant --- number of relevant documents among those.
            prob_of_doc_vert --- probability of a document to be a vertical document.
                                 prob_of_doc_vert == [0.1, 0.3, 0.2] means that each
                                 document belongs to Vert1 with probability 0.1,
                                 belongs to Vert2 with probability 0.3,
                                 to Vert3 with probability 0.2,
                                 and to Web with probability 0.4 (1 - 0.1 - 0.3 - 0.2).
            vert_rel --- specifies if the vertical documents are "non-relevant", "all-relevant"
                or just as "relevant" as the non-vertical documents.

            Returns: list of documents and the labels:
                     labels[doc.get_id()] == 1 <=> doc is relevant
        """
        if prob_of_doc_vert is None:
            prob_of_doc_vert = []
        if verticals is None:
            verticals = VERTICALS
        if num_relevant == "random":
            num_relevant = random.randint(1, num_documents // 2)
        elif type(num_relevant) == str and "-" in num_relevant:
            min_rel, max_rel = num_relevant.split("-")
            num_relevant = random.randint(int(min_rel), int(max_rel))
        else:
            num_relevant = int(num_relevant)

        assert num_documents > 0
        assert num_relevant > 0
        assert num_relevant <= num_documents
        ranking_with_vert_assignment = []
        for doc in range(num_documents):
            r = random.random()
            s = 0
            vert_index = 0
            for vert_index, prob in enumerate(prob_of_doc_vert):
                s += prob
                if s > r:
                    ranking_with_vert_assignment.append(
                        Document(doc, verticals[vert_index]))
                    break
            else:
                # non-vertical
                ranking_with_vert_assignment.append(Document(doc, "Web"))
        labels = [0] * num_documents
        documents_with_random_relevance = range(num_documents)
        if vert_rel != "relevant":  # "non-relevant" or "all-relevant"
            # Assign relevance for vertical and non-vertical documents separately
            documents_with_random_relevance = [id for id in xrange(num_documents) if \
                    ranking_with_vert_assignment[id].get_type() == "Web"]
        for id in random.sample(documents_with_random_relevance, num_relevant):
            labels[id] = 1
        if vert_rel == "all-relevant":
            for id in xrange(num_relevant):
                if ranking_with_vert_assignment[id].get_type() != "Web":
                    labels[id] = 1
        return ranking_with_vert_assignment, labels

    @staticmethod
    def _generate_ranking(base_documents, serp_size=10, decay_power=5):
        # Select documents from the base_documents list
        # with decreasing softmax probability.
        probs = [1.0 / (i + 1) ** decay_power for i in range(len(base_documents))]
        # create a copy
        docs = [d for d in base_documents]
        s = sum(probs)
        normalize(probs)
        ranking = []
        for iteration_number in range(serp_size):
            r = random.random()
            s = 0
            for pos, prob in enumerate(probs):
                s += prob
                if s >= r:
                    break
            ranking.append(docs[pos])
            # Exclude the docid that we've just used.
            probs = [prob for pos1, prob in enumerate(probs) if pos1 != pos]
            docs = [rank for pos1, rank in enumerate(docs) if pos1 != pos]
            normalize(probs)
            # assert 0.95 <= sum(probs) <= 1.05
        # Now re-order the documents to ensure that the vertical ones are grouped.
        reordered_ranking = []
        formed_blocks = set()
        for doc in ranking:
            if doc.get_type() == 0:
                reordered_ranking.append(doc)
            elif doc.get_type() not in formed_blocks:
                reordered_ranking += (
                        [d for d in ranking if d.get_type() == doc.get_type()])
                formed_blocks.add(doc.get_type())
        return reordered_ranking

    def _pareto_dominates(self, a, b, labels, orientation=None):
        # Cut and sort by exmaniation probability accoridng to user model
        a = a[:self.length]
        b = b[:self.length]
        # Here we rely on implicit orientation values (1.0), since we don't have
        # them anyway for synthetic data.
        examination_a = self.pareto_um.get_examination_prob(a, orientation=orientation)
        examination_b = self.pareto_um.get_examination_prob(b, orientation=orientation)
        a.sort(key=dict(zip(a, examination_a)).get, reverse=True)
        b.sort(key=dict(zip(b, examination_b)).get, reverse=True)

        rel_a = [index for index, item in enumerate(a) if labels[item.get_id()] > 0]
        rel_b = [index for index, item in enumerate(b) if labels[item.get_id()] > 0]
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
    def get_online_metrics(clicks, ranking):
        if all(c == 0 for c in clicks):
            return {
                "MaxRR":        0.0,
                "MinRR":        0.0,
                "MeanRR":       0.0,
                "PLC":          0.0,
                "Clicks@1":     0.0,
                "VertClick":    0.0,
                "click_counts": 0.0,
            }
        else:
            scores = {}
            minClickRank = min((r for r, c in enumerate(clicks) if c))
            maxClickRank = max((r for r, c in enumerate(clicks) if c))
            numClicks = sum(1 for c in clicks if c)
            scores["MaxRR"]     = 1.0 / (minClickRank + 1)
            scores["MinRR"]     = 1.0 / (maxClickRank + 1)
            scores["MeanRR"]    = (
                sum((1.0 / (r + 1) for r, c in enumerate(clicks) if c)) / numClicks
            )
            scores["PLC"]       = float(numClicks) / (maxClickRank + 1)
            scores["Clicks@1"]  = 1.0 if clicks[0] else 0.0
            scores["VertClick"] = (
                1.0 if any(
                    (c for (d, c) in zip(ranking, clicks) if d.get_type() != "Web")
                ) else 0.0
            )
            scores["click_counts"] = sum(1 for c in clicks if c)
            return scores
