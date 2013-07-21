# KH, 2012/06/19

import argparse

from numpy import asarray, where
from random import randint
from utils import split_arg_str

from AbstractInterleavedComparison import AbstractInterleavedComparison


class BalancedInterleave(AbstractInterleavedComparison):
    """Interleave and compare rankers using the original balanced
    interleave method."""

    def __init__(self, arg_str="random"):
        if arg_str.startswith("--"):
            parser = argparse.ArgumentParser(description="Parse arguments for "
                "interleaving method.", prog=self.__class__.__name__)
            parser.add_argument("-s", "--startinglist")
            args = vars(parser.parse_known_args(split_arg_str(arg_str))[0])
            if "startinglist" in args:
                self.startinglist = args["startinglist"]
            else:
                self.startinglist = "random"
        else:
            self.startinglist = arg_str

    def interleave(self, r1, r2, query, length):
        # get ranked list for each ranker (put in assignment var)
        l1, l2 = [], []
        r1.init_ranking(query)
        r2.init_ranking(query)
        length = min(r1.document_count(), r2.document_count(), length)
        for _ in range(length):
            l1.append(r1.next())
            l2.append(r2.next())
        # interleave
        l = []
        i1, i2 = 0, 0

        if self.startinglist == "random":
            # pick starting list at random
            first = randint(0, 1)
        elif self.startinglist == "fixed":
            first = 0
        else:
            raise Exception("Unknown starting method '%s' for "
                            "comparison method %s." %
                            (self.startinglist, self.__class__.__name__))

        # interleave deterministically
        while len(l) < length:
            if (i1 < i2) or (i1 == i2 and first == 0):
                if l1[i1] not in l:
                    l.append(l1[i1])
                i1 += 1
            else:
                if l2[i2] not in l:
                    l.append(l2[i2])
                i2 += 1
        # for balanced interleave the assignment captures the two original
        # ranked result lists l1 and l2
        return (asarray(l), (asarray(l1), asarray(l2)))

    def infer_outcome(self, l, a, c, query):
        c = asarray(c)
        a = (asarray(a[0]), asarray(a[1]))
        click_ids = where(c == 1)[0]
        if not len(click_ids):  # no clicks, will be a tie
            return 0
        # find minimum rank of the lowest click: k
        clicks_on_l1 = []
        clicks_on_l2 = []
        for clicked in click_ids:
            a1_clicks = where(a[0] == l[clicked])
            if len(a1_clicks[0]):
                clicks_on_l1.append(a1_clicks[0][0])
            a2_clicks = where(a[1] == l[clicked])
            if len(a2_clicks[0]):
                clicks_on_l2.append(a2_clicks[0][0])
        # lowest click
        lowest_click = -1
        if len(clicks_on_l1) and len(clicks_on_l2):
            lowest_click = min(max(clicks_on_l1), max(clicks_on_l2))
        elif len(clicks_on_l1):
            lowest_click = max(clicks_on_l1)
        elif len(clicks_on_l2):
            lowest_click = max(clicks_on_l2)
        # get number of clicked documents ranked higher or equal to N
        # for both lists
        c1, c2 = 0, 0
        for i in click_ids:
            if where(a[0] == l[i]) <= lowest_click:
                c1 += 1
            if where(a[1] == l[i]) <= lowest_click:
                c2 += 1
        # compare and return outcome
        return -1 if c1 > c2 else 1 if c2 > c1 else 0
