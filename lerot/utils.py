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

# KH, 2012/06/20
"""
Utility functions
"""

from importlib import import_module
from numpy import dot, sqrt
import numpy as np
from scipy.linalg import norm
from random import sample


def string_to_boolean(string):
    string = string.lower()
    if string in ['0', 'f', 'false', 'no', 'off']:
        return False
    elif string in ['1', 't', 'true', 'yes', 'on']:
        return True
    else:
        raise ValueError()


def get_class(name):
    """Dynamically import lerot.<name>.

    Here be dragons.
    """
    module, classname = name.rsplit(".", 1)
    module = "lerot." + module

    try:
        return getattr(import_module(module), classname)
    except AttributeError as e:
        msg = ('%s while trying to import %r from %r'
               % (e.args[0], classname, module))
        e.args = (msg,) + e.args[1:]
        raise


def split_arg_str(arg_str):
    s = []
    max_index = 0
    while max_index < len(arg_str):
        index = arg_str.find("\"", max_index)
        # no more quotes: split the remaining args
        if index == -1:
            s.extend(arg_str[max_index:].split())
            break
        # quote found: find end + add preceding and quoted elements
        else:
            if index > max_index:
                s.extend(arg_str[max_index:index].split())
            closing_index = arg_str.find("\"", index + 1)
            if closing_index == -1:
                raise ValueError("Argument string contains non-matched quotes:"
                    " %s" % arg_str)
            s.append(arg_str[index + 1:closing_index])
            max_index = closing_index + 1
    return s


def rank(x, ties, reverse=False):
    n = len(x)
    if ties == "first":
        ix = zip(x, reversed(range(n)), range(n))
    elif ties == "last":
        ix = zip(x, range(n), range(n))
    elif ties == "random":
        ix = zip(x, sample(range(n), n), range(n))
    else:
        raise Exception("Unknown method for breaking ties: \"%s\"" % ties)
    ix.sort(reverse=reverse)
    indexes = [i for _, _, i in ix]
    return [i for _, i in sorted(zip(indexes, range(n)))]


def get_cosine_similarity(v1, v2):
    """Compute the cosine similarity between two vectors."""
    if norm(v1) == 0 or norm(v2) == 0:
        return 0.0
    return dot(v1, v2) / (norm(v1) * norm(v2))


def get_binomial_ci(p_hat, n):
    """Compute the binomial (Wilson) confidence interval for a given proportion
    estimate (p_hat) and sample size (n)."""
    # z for p=0.05/2 (97.5)
    # http://www.wepapers.com/Papers/17864/Percentiles_of_the_Standard_Normal_
    # Distribution
    zA = 1.960
    # http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    # http://www.evanmiller.org/how-not-to-sort-by-average-rating.html
        #n = float(totalW1 + totalW2)
    lower = (p_hat + zA * zA / (2 * n) - zA * sqrt((p_hat * (1 - p_hat) +
        zA * zA / (4 * n)) / n)) / (1 + zA * zA / n)
    upper = (p_hat + zA * zA / (2 * n) + zA * sqrt((p_hat * (1 - p_hat) +
        zA * zA / (4 * n)) / n)) / (1 + zA * zA / n)
    return (lower, upper)


def sample_unit_sphere(n):
    """See http://mathoverflow.net/questions/24688/efficiently-sampling-
    points-uniformly-from-the-surface-of-an-n-sphere"""
    v = np.random.randn(n)
    v /= norm(v)
    return v


def sample_fixed(self, n):
    return np.ones(n) / sqrt(n)
