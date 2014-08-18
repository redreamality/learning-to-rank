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

import os
import sys
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(basedir,'lerot'))
from math import sqrt
from mapping import mapping

def get_significance(mean_1, mean_2, std_1, std_2, n):
    significance = ""
    ste_1 = std_1 / sqrt(n)
    ste_2 = std_2 / sqrt(n)
    t = (mean_1 - mean_2) / sqrt(ste_1 ** 2 + ste_2 ** 2)
    if mean_1 > mean_2:
        # treatment is worse than baseline
        # values used are for 120 degrees of freedom
        # (http://changingminds.org/explanations/research/analysis/
        # t-test_table.htm)
        if abs(t) >= 2.62:
            significance = "\dubbelneer"
        elif abs(t) >= 1.98:
            significance = "\enkelneer"
    else:
        if abs(t) >= 2.62:
            significance = "\dubbelop"
        elif abs(t) >= 1.98:
            significance = "\enkelop"
    return significance

def get_weights(filename):
    if filename.endswith(".gz"):
        fh = gzip.open(filename, "r")
    else:
        fh = open(filename, "r")
    yamldata = yaml.load(fh, Loader=Loader)
    fh.close()
    if not yamldata or not "final_weights" in yamldata:
        return (0, [])

    return (yamldata["offline_ndcg"][-1], yamldata["final_weights"])


def get(l, onoff, measure):
    if len(l) == 9:
        if onoff == "online":
            _, _, _, _, _, mean, std, min, max = l
        else:
            _, mean, std, min, max, _, _, _, _ = l
    elif len(l) == 5:
        if onoff == "online":
            _, _, _, mean, std = l
        else:
            _, mean, std, _, _ = l
    if measure == "mean":
        return mean
    elif measure == "std":
        return std
    elif measure == "min":
        return min
    elif measure == "max":
        return max
