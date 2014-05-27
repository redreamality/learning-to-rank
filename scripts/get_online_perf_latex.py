#!/usr/bin/env python

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

from include import *
from math import sqrt


def get_significance(mean_1, mean_2, std_1, std_2, n):
    significance = ""
    ste_1 = std_1 / sqrt(n)
    ste_2 = std_2 / sqrt(n)
    t = (mean_1 - mean_2) / sqrt(ste_1 ** 2 + ste_2 ** 2)
    #print t
    if mean_1 > mean_2:  # treatment is worse than baseline
        # values used are for 120 degrees of freedom (http://changingminds.org/
        # explanations/research/analysis/t-test_table.htm)
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

def get_percent_improvement(mean_1, mean_2):
    return (mean_2 - mean_1) / mean_1 * 100.0

perf = []
max_perf = 0
n = 125
baseline = 1 # usually: 0 for BI, 1 for TD

files = sys.argv[1:]
for input_file in files:
    fh = open(input_file, "r")
    lines = fh.readlines()
    last_line = lines[-1]
    (_, _, _, mean, std) = last_line.split()
    mean = float(mean)
    std = float(std)
    if mean > max_perf:
        max_perf = mean
    perf.append((mean, std))

for i in range(len(perf)):
    significance = get_significance(perf[baseline][0], perf[i][0], perf[baseline][1],
        perf[i][1], n)
    if perf[i][0] == max_perf:
        print " & \\textbf{%.2f} & %s" % (perf[i][0], significance),
    else:
        print " & %.2f & %s" % (perf[i][0], significance),
print "%% baseline vs. last 2: %.1f%% %.1f%%" % (get_percent_improvement(perf[baseline][0], perf[-2][0]), get_percent_improvement(perf[baseline][0], perf[-1][0]))
#print ""

