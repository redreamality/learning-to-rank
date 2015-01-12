import argparse
from collections import defaultdict
from random import randint

from numpy import asarray, e, log, where

from AbstractInterleavedComparison import AbstractInterleavedComparison
from ..utils import split_arg_str


class ProbabilisticMultileave(AbstractInterleavedComparison):
    """Probabilistic ..."""

    def __init__(self, arg_str=None):
        '''
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
        ARGS:
        - rankers: a list of ...
        - query: ...
        - length: the desired length of the lists
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
        return (asarray(l), (a, rankers))

    def infer_outcome(self, l, a, c, query):
        # TODO: this is not implemented yet, only copied from Interleave
        raise Exception("Not implemented")
        (td_a, r1, r2) = a

        # for comparisons with TD, use naive comparison
        if self.compare_td:
            c1 = sum([1 if val_a == 0 and val_c == 1 else 0
                for val_a, val_c in zip(td_a, c)])
            c2 = sum([1 if val_a == 1 and val_c == 1 else 0
                for val_a, val_c in zip(td_a, c)])
            return -1 if c1 > c2 else 1 if c2 > c1 else 0

        # comparison with marginalization
        # are there any clicks? (otherwise it's a tie)
        click_ids = where(asarray(c) == 1)
        if not len(click_ids[0]):  # no clicks, will be a tie
            return 0

        r1.init_ranking(query)
        r2.init_ranking(query)

        # enumerate all possible assignments that go with l, add their
        # outcomes weighted by probabilities
        # original outcome is not needed in this case, only clicks and probs
        root = SimpleBinaryTree(None, 0.0, 0)  # root
        nextLevel = [root]
        currentLevel = []

        # traverse possible assignments breath-first
        log_p_a = len(l) * log(0.5)
        log_p_l = len(l) * log(0.5)

        for n in range(len(l)):
            currentLevel = nextLevel
            nextLevel = []
            p_r1 = r1.get_document_probability(l[n])
            p_r2 = r2.get_document_probability(l[n])
            # zero probability: observed list is not possible (e.g., with
            # deterministic rankers and historical data)
            if p_r1 == 0 and p_r2 == 0:
                return .0
            r1.rm_document(l[n])
            try:
                r2.rm_document(l[n])
            except:
                pass
            log_p_l += log(p_r1 + p_r2)

            for node in currentLevel:
                # expand children and add new nodes to queue for nextLevel
                # left child: r1 is selected, only expand if p > 0
                if p_r1 > 0:
                    p_left = node.prob + log(0.5 * p_r1)
                    o_left = node.outcome
                    if c[n] == 1:
                        o_left += -1
                    node.left = SimpleBinaryTree(node, p_left, o_left)
                    nextLevel.append(node.left)
                # right child: r2 is selected, only expand if p > 0
                if p_r2 > 0:
                    p_right = node.prob + log(0.5 * p_r2)
                    o_right = node.outcome
                    if c[n] == 1:
                        o_right += 1
                    node.right = SimpleBinaryTree(node, p_right, o_right)
                    nextLevel.append(node.right)
        # we have all log probabilities for outcomes =! 0
        o1 = 0.0
        o2 = 0.0
        for node in nextLevel:
            if node.outcome != 0:
                # log_p_a and log_p_l cancel out if we turn the outcome into a
                # ratio for now, keep them for clarity
                if node.outcome < 0:
                    o1 += e ** (node.prob + log_p_a - log_p_l)
                else:
                    o2 += e ** (node.prob + log_p_a - log_p_l)

        # return -1 if o1 > o2 else 1 if o2 > o1 else 0
        if o1 == o2:
            outcome = 0
        elif self.aggregate == "expectation":
            outcome = o2 - o1
        elif self.aggregate == "log-likelihood-ratio":
            if o1 > o2:
                outcome = log(o2 / o1)
            else:
                outcome = log(o1 / o2)
        elif self.aggregate == "likelihood-ratio":
            if o1 > o2:
                outcome = (float(o2) / o1) - 1
            else:
                outcome = 1 - (float(o1) / o2)
        elif self.aggregate == "log-ratio":
            if o1 > o2:
                outcome = float(log(o1)) / log(o2) - 1
            else:
                outcome = 1 - float(log(o2)) / log(o1)
        elif self.aggregate == "binary":
            outcome = -1 if o1 > o2 else 1 if o2 > o1 else 0
        else:
            raise ValueError("Unknown aggregation method: %s", self.aggregate)
        return outcome

    def get_probability_of_list(self, result_list, context, query):
        # TODO: this is not implemented yet, only copied from Interleave
        raise Exception("Not implemented")

        # P(l) = \prod_{doc in result_list} 1/2 P_1(doc) + 1/2 P_2(doc)
        p_l = 1.0
        (_, r1, r2) = context
        r1.init_ranking(query)
        r2.init_ranking(query)
        for _, doc in enumerate(result_list):
            p_r1 = r1.get_document_probability(doc)
            p_r2 = r2.get_document_probability(doc)
            r1.rm_document(doc)
            r2.rm_document(doc)
            p_l *= 0.5 * (p_r1 + p_r2)
        return p_l


class SimpleBinaryTree:
    """tree that keeps track of outcome, probability of arriving at this
    outcome"""
    parent, left, right, prob, outcome = None, None, None, 0.0, 0

    def __init__(self, parent, prob, outcome):
        self.parent = parent
        self.prob = prob
        self.outcome = outcome
