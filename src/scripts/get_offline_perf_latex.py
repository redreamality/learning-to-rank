#!/usr/bin/env python
from include import *
from math import sqrt


def get_significance(mean_1, mean_2, std_1, std_2, n):
    significance = ""
    ste_1 = std_1 / sqrt(n)
    ste_2 = std_2 / sqrt(n)
    t = (mean_1 - mean_2) / sqrt(ste_1 ** 2 + ste_2 ** 2)
    #print t
    if mean_1 > mean_2:  # treatment is worse than baseline
        # values used are for 120 degrees of freedom #(http://changingminds.org
        # /explanations/research/analysis/t-test_table.htm)
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


perf = []
max_perf = 0
n = 125

files = sys.argv[1:]
for input_file in files:
    fh = open(input_file, "r")
    lines = fh.readlines()
    last_line = lines[-1]
    (_, mean, std, _, _) = last_line.split()
    mean = float(mean)
    std = float(std)
    if mean > max_perf:
        max_perf = mean
    perf.append((mean, std))

for i in range(len(perf)):
    significance = get_significance(perf[0][0], perf[i][0], perf[0][1],
        perf[i][1], n)
    if perf[i][0] == max_perf:
        print " & \\textbf{%.3f} & %s" % (perf[i][0], significance),
    else:
        print " & %.3f & %s" % (perf[i][0], significance),
#print ""
