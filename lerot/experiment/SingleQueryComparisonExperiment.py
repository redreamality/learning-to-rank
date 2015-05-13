# KH, 2012/09/12
# Runs a comparison experiment with a single query.

import argparse
import logging
import os.path
import yaml

from numpy import asarray, log, mean, where, zeros
from random import choice, randint
from scipy.stats.stats import kendalltau

from ..utils import get_class
from ..evaluation import NdcgEval
from ..query import load_queries


class SingleQueryComparisonExperiment():
    """Represents an experiment in which rankers are compared using interleaved
    comparisons on a single query.
    """

    def __init__(self, query_dir, feature_count, log_fh, args):
        """Initialize an experiment using the provided arguments."""
        self.log_fh = log_fh
        self.query_dir = query_dir
        self.feature_count = feature_count
        self.ties = "first"
        # randomly select a query file, and load a single query from that file
        count_attempts = 0
        while not hasattr(self, "query"):
            count_attempts += 1
            if count_attempts > 100:
                raise ValueError("Did not find a query with more than one "
                    "relevance levels after 100 attempts.")
            query_file = os.path.join(query_dir, choice(os.listdir(query_dir)))
            logging.info("Loading query %s." % query_file)
            query = load_queries(query_file, feature_count).values()[0]
            if "check_queries" in args and args["check_queries"]:
                # make sure that there's at least two different relevance
                # grades
                if len(query.get_labels()) > 1 and len(where(asarray(
                    query.get_labels()) != query.get_labels()[0])[0]) > 0:
                    self.query = query
                else:
                    continue
            else:
                self.query = query
        self.query_id = self.query.get_qid()
        # sample source and target ranker pair, check that there is an ndcg
        # difference if needed
        count_attempts = 0
        while not hasattr(self, "source_pair"):
            count_attempts += 1
            if count_attempts > 100:
                raise ValueError("Did not find ranker pairs \w ndcg diff != 0 "
                    "relevance levels after 100 attempts.")
            src_1 = self._sample_ranker_without_replacement(
                self.feature_count, [])
            src_2 = self._sample_ranker_without_replacement(self.feature_count,
                [src_1])
            tar_1 = self._sample_ranker_without_replacement(self.feature_count,
                [src_1, src_2])
            tar_2 = self._sample_ranker_without_replacement(self.feature_count,
                [tar_1, src_1, src_2])
            # check arguments: if needed - make sure there is a difference
            # between the _target_ rankers
            prob_src_rankers = self._get_ranker_pair(
                "ranker.ProbabilisticRankingFunction",
                [3], (src_1, src_2), self.feature_count, self.ties)
            prob_tar_rankers = self._get_ranker_pair(
                "ranker.ProbabilisticRankingFunction",
                [3], (tar_1, tar_2), self.feature_count, self.ties)
            ndcg = NdcgEval()
            o1 = ndcg.evaluate_one(prob_tar_rankers[0], self.query, -1,
                ties=self.ties)
            o2 = ndcg.evaluate_one(prob_tar_rankers[1], self.query, -1,
                ties=self.ties)
            ndcg_diff = float(o2 - o1)
            if "check_rankers" in args and args["check_rankers"] and \
                ndcg_diff == 0:
                continue
            self.source_pair = [src_1, src_2]
            self.target_pair = [tar_1, tar_2]
            self.ndcg_diff = ndcg_diff
            # compute stats for the ranker pairs
            #most_likely_source_list = self._get_most_likely_list(
            #    prob_src_rankers[0], prob_src_rankers[1], self.query)
            #most_likely_target_list = self._get_most_likely_list(
            #    prob_tar_rankers[0], prob_tar_rankers[1], self.query)
            #k = kendalltau(most_likely_source_list, most_likely_target_list)
            #if isinstance(k, tuple):
            #    k = k[0]
            #self.kendalltau = float(k)
            (_, combined_source_dist) = \
                self._get_combined_document_distribution(prob_src_rankers[0],
                    prob_src_rankers[1], self.query)
            (_, combined_target_dist) = \
                self._get_combined_document_distribution(prob_tar_rankers[0],
                    prob_tar_rankers[1], self.query)
            self.kullback_leibler_src = float(
                self._get_kullback_leibler_divergence(combined_source_dist,
                                                      combined_target_dist))
            self.kullback_leibler_tar = float(
                self._get_kullback_leibler_divergence(combined_target_dist,
                                                      combined_source_dist))
            self.jensen_shannon = float(self._get_jensen_shannon_divergence(
                combined_source_dist, combined_target_dist))
            self.l1_norm = float(self._get_l1_norm(combined_source_dist,
                combined_target_dist))
        # construct experiment according to provided arguments
        self.result_length = args["result_length"]
        self.num_queries = args["num_queries"]
        self.um_class = get_class(args["user_model"])
        self.um_args = args["user_model_args"]
        self.um = self.um_class(self.um_args)
        # set up methods to compare
        parser = argparse.ArgumentParser(description="parse arguments of an "
            "evaluation method.", prog="evaluation method configuration")
        parser.add_argument("-c", "--class_name")
        parser.add_argument("-r", "--ranker")
        parser.add_argument("-a", "--ranker_args")
        parser.add_argument("-s", "--source_ranker")
        parser.add_argument("-b", "--source_ranker_args")
        parser.add_argument("-i", "--interleave_method")

        self.target_rankers = {}
        self.source_rankers = {}
        self.live_methods = {}
        self.hist_methods = {}
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
                if not ranker in self.target_rankers:
                    self.target_rankers[ranker] = {}
                if not ranker_args in self.target_rankers[ranker]:
                    self.target_rankers[ranker][ranker_args] = {}
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
                self.hist_methods[method]["interleave_method"] = \
                get_class(method_args["interleave_method"])()
                # init (target) rankers
                ranker = method_args["ranker"]
                ranker_args = method_args["ranker_args"]
                self.hist_methods[method]["ranker"] = ranker
                self.hist_methods[method]["ranker_args"] = ranker_args
                if not ranker in self.target_rankers:
                    self.target_rankers[ranker] = {}
                if not ranker_args in self.target_rankers[ranker]:
                    self.target_rankers[ranker][ranker_args] = {}
                # init (source) rankers
                if method_args["source_ranker"]:
                    source_ranker = method_args["source_ranker"]
                    source_ranker_args = method_args["source_ranker_args"]
                else:
                    source_ranker = method_args["ranker"]
                    source_ranker_args = method_args["ranker_args"]
                self.hist_methods[method]["source_ranker"] = source_ranker
                self.hist_methods[method]["source_ranker_args"] = \
                    source_ranker_args
                if not source_ranker in self.source_rankers:
                    self.source_rankers[source_ranker] = {}
                if not source_ranker_args in self.source_rankers[
                    source_ranker]:
                    self.source_rankers[source_ranker][source_ranker_args] = {}
        # init target rankers needed by live and/or hist methods
        for ranker in self.target_rankers:
            for ranker_args in self.target_rankers[ranker]:
                self.target_rankers[ranker][ranker_args] = \
                    self._get_ranker_pair(ranker, ranker_args,
                    self.target_pair, self.feature_count, self.ties)
        # init source rankers needed by hist methods
        for ranker in self.source_rankers:
            for ranker_args in self.source_rankers[ranker]:
                self.source_rankers[ranker][ranker_args] = \
                    self._get_ranker_pair(ranker, ranker_args,
                    self.source_pair, self.feature_count, self.ties)

    def _sample_ranker_without_replacement(self, num_features, exclude):
        while True:
            feature = randint(0, num_features - 1)
            if feature not in exclude:
                return feature

    def _get_weight_vector(self, feature_id, num_features):
        weights = zeros(num_features)
        weights[feature_id] = 1.
        return weights

    def _get_ranker_pair(self, class_name, ranker_args, feature_pair,
        feature_count, ties):
        
        return (get_class(class_name)(ranker_args, ties, feature_count,
                init=self._get_weight_vector(feature_pair[0], feature_count)),
            get_class(class_name)(ranker_args, ties, feature_count,
                init=self._get_weight_vector(feature_pair[0], feature_count)))

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
        # initialize counts and outcome arrays
        query_ids = []
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
        for _ in range(self.num_queries):
            query_ids.append(self.query_id)
            # apply live methods (use target rankers only)
            for method_id, method in self.live_methods.items():
                target_pair = self.target_rankers[method["ranker"]][
                    method["ranker_args"]]
                (l, a) = method["instance"].interleave(target_pair[0],
                    target_pair[1], self.query, self.result_length)
                clicks = self.um.get_clicks(l, self.query.get_labels())
                o = method["instance"].infer_outcome(l, a, clicks, self.query)
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
                    method["source_ranker"], method["source_ranker_args"])
                target_pair = self.target_rankers[method["ranker"]][
                    method["ranker_args"]]
                source_pair = self.source_rankers[method["source_ranker"]][
                    method["source_ranker_args"]]
                if not interleave_key in result_lists:
                    (l, a) = method["interleave_method"].interleave(
                        source_pair[0], source_pair[1],
                        self.query, self.result_length)
                    result_lists[interleave_key] = l
                    assignments[interleave_key] = a
                    clicks[interleave_key] = self.um.get_clicks(l,
                        self.query.get_labels())
                hist_click_counts[method_id].append(
                    clicks[interleave_key].tolist().count(1))

                o = method["instance"].infer_outcome(
                    result_lists[interleave_key], assignments[interleave_key],
                    clicks[interleave_key], target_pair[0], target_pair[1],
                    self.query)
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
            "ndcg_diffs": [self.ndcg_diff],
            # similarities between source and target pairs
            #"kendall_tau": [self.kendalltau],
            "kullback_leibler_src": [self.kullback_leibler_src],
            "kullback_leibler_tar": [self.kullback_leibler_tar],
            "jensen_shannon": [self.jensen_shannon],
            "l1_norm": [self.l1_norm]
            }, self.log_fh, default_flow_style=False)
