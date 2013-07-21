# KH 07/10/2012

import argparse
import re

from random import random
from numpy import zeros
from utils import split_arg_str

from AbstractUserModel import AbstractUserModel


class CascadeUserModel(AbstractUserModel):
    """Defines a cascade user model, simulating a user that inspects results
    starting from the top of a result list."""

    def __init__(self, arg_str):
        parser = argparse.ArgumentParser(description="Initialize a cascade "
            "user model with click and stop probabilities.",
            prog="CascadeUserModel")
        parser.add_argument("-c", "--p_click", nargs="+")
        parser.add_argument("-s", "--p_stop", nargs="+")
        args = vars(parser.parse_args(split_arg_str(arg_str)))
        # allow arbitrary hash maps to map relevance labels to click and stop
        # probabilities
        p_click_str = "".join(args["p_click"]).strip("\"")
        self.p_click = {}
        for entry in re.split("\s*,\s*", p_click_str):
            (key, value) = re.split("\s*:\s*", entry)
            self.p_click[int(key)] = float(value)
        self.p_stop = {}
        p_stop_str = "".join(args["p_stop"]).strip("\"")
        for entry in re.split("\s*,\s*", p_stop_str):
            (key, value) = re.split("\s*:\s*", entry)
            self.p_stop[int(key)] = float(value)

    def get_clicks(self, result_list, labels):
        """simulate clicks on list l"""
        c = zeros(len(result_list), dtype='int')
        for pos, r in enumerate(result_list):
            label = labels[r]
            if label not in self.p_click:
                raise Exception("Cardinality of labels does not match the user"
                                " model.")
            # generate a random number between 0 and 1
            rand = random()
            if rand < self.p_click[label]:
                c[pos] = 1  # click at rank r
                # if there was a click, determine whether to stop
                rand = random()
                if rand < self.p_stop[label]:
                    break
        return c
