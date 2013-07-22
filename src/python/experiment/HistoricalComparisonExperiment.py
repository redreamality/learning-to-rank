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

# KH, 2012/08/15
"""
Runs a comparison experiment with historical data reuse.
"""

import argparse
import random
import yaml

from numpy import log, mean, zeros
from scipy.stats import kendalltau
from utils import get_class

import evaluation


class HistoricalComparisonExperiment():
    """Represents an experiment in which rankers are compared using interleaved
    comparisons with live and historic click data.
    """

    def __init__(self, queries, feature_count, log_fh, args):
        """Initialize an experiment using the provided arguments."""
        self.log_fh = log_fh
        self.queries = queries
        self.feature_count = feature_count
        self.ties = "first"
        # construct experiment according to provided arguments
        self.result_length = args["result_length"]
        self.num_queries = args["num_queries"]
        self.query_sampling_method = args["query_sampling_method"]
        self.um_class = get_class(args["user_model"])
        self.um_args = args["user_model_args"]
        self.um = self.um_class(self.um_args)
        # set up methods to compare
        parser = argparse.ArgumentParser(description="parse arguments of an "
            "evaluation method.", prog="evaluation method configuration")
        parser.add_argument("-c", "--class_name")
        parser.add_argument("-r", "--ranker")
        parser.add_argument("-a", "--ranker_args")
        parser.add_argument("-i", "--interleave_method")
        self.rankers = {}
        self.live_methods = {}
        self.hist_methods = {}
        self.ndcg = evaluation.NdcgEval()
        # init live methods
        if "live_evaluation_methods" in args:
            for method_id, method in enumerate(
                    args["live_evaluation_methods"]):
                self.live_methods[method] = {}
                method_args_str = \
                    args["live_evaluation_methods_args"][method_id]
                method_args = vars(parser.parse_known_args(
                    method_args_str.split())[0])
                class_name = method_args["class_name"]
                self.live_methods[method]["instance"] = \
                    get_class(class_name)(method_args_str)
                ranker = method_args["ranker"]
                ranker_args = method_args["ranker_args"]
                self.live_methods[method]["ranker"] = ranker
                self.live_methods[method]["ranker_args"] = ranker_args
                if not ranker in self.rankers:
                    self.rankers[ranker] = {}
                if not ranker_args in self.rankers[ranker]:
                    self.rankers[ranker][ranker_args] = {}
        # init hist methods
        if "hist_evaluation_methods" in args:
            for method_id, method in enumerate(
                    args["hist_evaluation_methods"]):
                self.hist_methods[method] = {}
                method_args_str = \
                    args["hist_evaluation_methods_args"][method_id]
                method_args = vars(parser.parse_known_args(
                    method_args_str.split())[0])
                class_name = method_args["class_name"]
                self.hist_methods[method]["instance"] = \
                    get_class(class_name)(method_args_str)
                ranker = method_args["ranker"]
                ranker_args = method_args["ranker_args"]
                self.hist_methods[method]["ranker"] = method_args["ranker"]
                self.hist_methods[method]["ranker_args"] = \
                    method_args["ranker_args"]
                if not ranker in self.rankers:
                    self.rankers[ranker] = {}
                if not ranker_args in self.rankers[ranker]:
                    self.rankers[ranker][ranker_args] = {}
                self.hist_methods[method]["interleave_method"] = \
                get_class(method_args["interleave_method"])()
        # sample source and target ranker pair, create deterministic and
        # probabilistic ranker pairs
        self.source_pair = [0, 0]
        self.source_pair[0] = self._sample_ranker_without_replacement(
            self.feature_count, [])
        self.source_pair[1] = self._sample_ranker_without_replacement(
            self.feature_count, [self.source_pair[0]])
        self.target_pair = [0, 0]
        self.target_pair[0] = self._sample_ranker_without_replacement(
            self.feature_count, self.source_pair)
        self.target_pair[1] = self._sample_ranker_without_replacement(
            self.feature_count, [self.target_pair[0], self.source_pair[0],
            self.source_pair[1]])
        # init rankers needed by live and/or hist methods
        for ranker in self.rankers:
            for ranker_args in self.rankers[ranker]:
                self.rankers[ranker][ranker_args]["source"] = \
                    self._get_ranker_pair(ranker, ranker_args,
                    self.source_pair, self.feature_count, self.ties)
                self.rankers[ranker][ranker_args]["target"] = \
                    self._get_ranker_pair(ranker, ranker_args,
                    self.target_pair, self.feature_count, self.ties)

    def _sample_qid(self, query_keys, query_count, query_length):
            if self.query_sampling_method == "random":
                return query_keys[random.randint(0, query_length - 1)]
            elif self.query_sampling_method == "fixed":
                return query_keys[query_count % query_length]

    def _sample_ranker_without_replacement(self, num_features, exclude):
        while True:
            feature = random.randint(0, num_features - 1)
            if feature not in exclude:
                return feature

    def _get_weight_vector(self, feature_id, num_features):
        weights = zeros(num_features)
        weights[feature_id] = 1.
        return weights

    def _get_ranker_pair(self, class_name, ranker_args, feature_pair,
        feature_count, ties):
        return (get_class(class_name)(ranker_args, self._get_weight_vector(
                feature_pair[0], feature_count), ties=ties),
            get_class(class_name)(ranker_args, self._get_weight_vector(
                feature_pair[1], feature_count), ties=ties))

    def _get_most_likely_list(self, r1, r2, query):
        """get the most likely interleaved list for a given pair of rankers"""
        (docids, probs) = self._get_combined_document_distribution(r1, r2,
            query)
        tmp = [(prob, docid) for docid, prob in zip(docids, probs)]
        tmp.sort(reverse=True)
        l = [docid for prob, docid in tmp]
        return l

    def _get_combined_document_distribution(self, r1, r2, query):
        """get the distribution over documents given a pair of rankers"""
        r1.init_ranking(query)
        r2.init_ranking(query)
        docids = query.get_docids()
        probs = [r1.get_document_probability(docid) +
                 r2.get_document_probability(docid) for docid in docids]
        return (docids, probs)

    def _get_kullback_leibler_divergence(self, probs1, probs2):
        return sum([p1 * log(p1 / p2) for p1, p2 in zip(probs1, probs2)])

    def _get_jensen_shannon_divergence(self, probs1, probs2):
        mean_probs = [mean([p1, p2]) for p1, p2 in zip(probs1, probs2)]
        return float(
            .5 * self._get_kullback_leibler_divergence(probs1, mean_probs) +
            .5 * self._get_kullback_leibler_divergence(probs2, mean_probs))

    def _get_l1_norm(self, probs1, probs2):
        return sum([abs(p1 - p2) for p1, p2 in zip(probs1, probs2)])

    def run(self):
        """Run the experiment for num_queries queries."""
        query_keys = sorted(self.queries.keys())
        query_length = len(query_keys)
        # for bookkeeping
        query_ids = []
        ndcg_diffs = []
        per_query_kendalltau = []
        per_query_kullback_leibler_src = []
        per_query_kullback_leibler_tar = []
        per_query_jensen_shannon = []
        per_query_l1_norm = []
        prob_src_rankers = self._get_ranker_pair(
            "ranker.ProbabilisticRankingFunction",
            3, self.source_pair, self.feature_count, self.ties)
        prob_tar_rankers = self._get_ranker_pair(
            "ranker.ProbabilisticRankingFunction",
            3, self.target_pair, self.feature_count, self.ties)
        # initialize counts and outcome arrays
        live_outcomes = {}
        live_click_counts = {}
        for method_id in self.live_methods:
            live_outcomes[method_id] = []
            live_click_counts[method_id] = []
        hist_outcomes = {}
        hist_click_counts = {}
        for method_id in self.hist_methods:
            hist_outcomes[method_id] = []
            hist_click_counts[method_id] = []
        # process num_queries queries
        for query_count in range(self.num_queries):
            qid = self._sample_qid(query_keys, query_count, query_length)
            query_ids.append(qid)
            query = self.queries[qid]
            o1 = self.ndcg.evaluate_one(prob_tar_rankers[0].w, query, -1,
                ties=self.ties)
            o2 = self.ndcg.evaluate_one(prob_tar_rankers[1].w, query, -1,
                ties=self.ties)
            ndcg_diffs.append(float(o2 - o1))
            # compute similarities between ranker pairs (for probabilistic
            # rankers)
            most_likely_source_list = self._get_most_likely_list(
                prob_src_rankers[0], prob_src_rankers[1], query)
            most_likely_target_list = self._get_most_likely_list(
                prob_tar_rankers[0], prob_tar_rankers[1], query)
            k = kendalltau(most_likely_source_list, most_likely_target_list)
            if isinstance(k, tuple):
                k = k[0]
            per_query_kendalltau.append(float(k))
            (_, combined_source_dist) = \
                self._get_combined_document_distribution(prob_src_rankers[0],
                    prob_src_rankers[1], query)
            (_, combined_target_dist) = \
                self._get_combined_document_distribution(prob_tar_rankers[0],
                    prob_tar_rankers[1], query)
            per_query_kullback_leibler_src.append(float(
                self._get_kullback_leibler_divergence(combined_source_dist,
                                                      combined_target_dist)))
            per_query_kullback_leibler_tar.append(float(
                self._get_kullback_leibler_divergence(combined_target_dist,
                                                      combined_source_dist)))
            per_query_jensen_shannon.append(float(
                self._get_jensen_shannon_divergence(combined_source_dist,
                                                    combined_target_dist)))
            per_query_l1_norm.append(float(
                self._get_l1_norm(combined_source_dist,
                                        combined_target_dist)))
            # apply live methods (use target rankers only)
            for method_id, method in self.live_methods.items():
                ranker_pairs = \
                    self.rankers[method["ranker"]][method["ranker_args"]]
                (l, a) = method["instance"].interleave(
                    ranker_pairs["target"][0], ranker_pairs["target"][1],
                    query, self.result_length)
                clicks = self.um.get_clicks(l, query.get_labels())
                o = method["instance"].infer_outcome(l, a, clicks, query)
                live_outcomes[method_id].append(float(o))
                live_click_counts[method_id].append(clicks.tolist().count(1))
            # apply historical methods (use source rankers to collect data,
            # reuse collected data when possible)
            result_lists = {}
            assignments = {}
            clicks = {}
            for method_id, method in self.hist_methods.items():
                interleave_key = "%s-%s-%s" % (
                    method["interleave_method"].__class__.__name__,
                    method["ranker"], method["ranker_args"])
                ranker_pairs = \
                    self.rankers[method["ranker"]][method["ranker_args"]]
                if not interleave_key in result_lists:
                    (l, a) = method["interleave_method"].interleave(
                        ranker_pairs["source"][0], ranker_pairs["source"][1],
                        query, self.result_length)
                    result_lists[interleave_key] = l
                    assignments[interleave_key] = a
                    clicks[interleave_key] = self.um.get_clicks(l,
                        query.get_labels())
                hist_click_counts[method_id].append(
                    clicks[interleave_key].tolist().count(1))

                o = method["instance"].infer_outcome(
                    result_lists[interleave_key],
                    assignments[interleave_key],
                    clicks[interleave_key],
                    ranker_pairs["target"][0], ranker_pairs["target"][1],
                    query)
                hist_outcomes[method_id].append(float(o))
        # record ranker pairs, comparison outcomes
        yaml.dump({
            "source_pair": self.source_pair,
            "target_pair": self.target_pair,
            "live_outcomes": live_outcomes,
            "hist_outcomes": hist_outcomes,
            "live_click_counts": live_click_counts,
            "hist_click_counts": hist_click_counts,
            "query_ids": query_ids,
            # ndcg difference between the target rankers
            "ndcg_diffs": ndcg_diffs,
            # similarities between source and target pairs
            "kendall_tau": per_query_kendalltau,
            "kullback_leibler_src": per_query_kullback_leibler_src,
            "kullback_leibler_tar": per_query_kullback_leibler_tar,
            "jensen_shannon": per_query_jensen_shannon,
            "l1_norm": per_query_l1_norm
            }, self.log_fh, default_flow_style=False)
