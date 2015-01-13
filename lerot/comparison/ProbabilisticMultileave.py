import argparse
from collections import defaultdict
from random import randint

from numpy import asarray, e, log, where

from AbstractInterleavedComparison import AbstractInterleavedComparison
from ..utils import split_arg_str
from nltk.test import probability_fixt


class ProbabilisticMultileave(AbstractInterleavedComparison):
    """Probabilistic ..."""

    def __init__(self, arg_str=None):
        '''
        TODO: docstring

        ARGS:
        - aggregate: "expectation", "log-likelihood-ratio",
            "likelihood-ratio", "log-ratio", "binary"
        - det_interleave: If true, use deterministic interleaving, regardless
            of the ranker type used for comparison.
        - compare_td: if TRUE, compare rankers using observed assignment
            instead of marginalizing over possible assignments.
        '''
        if arg_str:
            parser = argparse.ArgumentParser(description="Parse arguments for "
                "interleaving method.", prog=self.__class__.__name__)
            parser.add_argument("-a", "--aggregate", choices=["expectation",
                "log-likelihood-ratio", "likelihood-ratio", "log-ratio",
                "binary"])
            parser.add_argument("-d", "--det_interleave", type=bool,
                help="If true, use deterministic interleaving, regardless "
                "of the ranker type used for comparison.")
            parser.add_argument("-t", "--compare_td", type=bool,
                help="If true, compare rankers using observed assignments "
                "instead of marginalizing over possible assignments.")
            args = vars(parser.parse_known_args(split_arg_str(arg_str))[0])
            if "aggregate" in args and args["aggregate"]:
                self.aggregate = args["aggregate"]
            if "det_interleave" in args and args["det_interleave"]:
                self.det_interleave = True
            if "compare_td" in args and args["compare_td"]:
                self.compare_td = True
        if not hasattr(self, "aggregate") or not self.aggregate:
            self.aggregate = "expectation"
        if not hasattr(self, "det_interleave"):
            self.det_interleave = False
        if not hasattr(self, "compare_td"):
            self.compare_td = False

    def multileave(self, rankers, query, length):
        '''
        TODO: DOCSTRING

        ARGS:
        - rankers: a list of ...
        - query: ...
        - length: the desired length of the lists

         RETURNS:
        - l: multileaved list of documents
        - a: list indicating which ranker is used at each row of l
        '''
        d = defaultdict(list)
        for i, r in enumerate(rankers):
            d[i] = r.init_ranking(query)

        length = min([length] + [r.document_count() for r in rankers])
        # start with empty document list
        l = []
        # random bits indicate which r to use at each rank
        a = asarray([randint(0, len(rankers) - 1) for _ in range(length)])
        for next_a in a:
            # flip coin - which r contributes doc (pre-computed in a)
            select = rankers[next_a]
            others = [r for r in rankers if r is not select]
            # draw doc
            # TODO add deterministic interleaving?
            pick = select.next()
            l.append(pick)
            # let other ranker know that we removed this document
            for o in others:
                try:
                    o.rm_document(pick)
                    # TODO remove try / catch block
                except:
                    pass
        return (asarray(l), a)

    def infer_outcome(self, l, rankers, c, query):
        '''
        TODO: DOCSTRING

        ARGS:
        - l: the created list of documents, using multileaving
        - rankers: the rankers
        - c: the clicks
        - query: the query

        RETURNS
        - The Credits
        '''

        click_ids = where(asarray(c) == 1)
        if not len(click_ids[0]):  # no clicks, will be a tie
            return 0

        rankers = [r.init_ranking(query) for r in rankers]
        p = self.probability_of_list(l, rankers, click_ids)
        credits = self.credits_of_list(p)

        return credits

    def probability_of_list(self, result_list, rankers, clickedDocs):
        '''
        ARGS:
        - result_list: the multileaved list
        - rankers: a list of rankers
        - clickedDocs: the docIds in the result_list which recieved a click

        RETURNS
        -p: list with the probability that the list comes from each ranker
        '''
        # TODO: this is not implemented yet, only copied from Interleave
        p = None
        return p

    def credits_of_list(self, p):
        '''
        ARGS:
        -p: list with the probability that the list comes from each ranker

        RETURNS:
        - credits: list of credits for each ranker
        '''
        credits = None
        return credits


class SimpleNAryTree:
    """TODO: this is not a tree! this is a node...

    tree that keeps track of outcome, probability of arriving at this
    outcome"""
    parent, left, right, prob, outcome = None, None, None, 0.0, 0

    def __init__(self, parent, prob, outcome):
        self.parent = parent
        self.prob = prob
        self.outcome = outcome
