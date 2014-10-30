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

# KH, 2012/06/25
"""
Summarizes the output of an online learning experiment. 
"""

try:
    from include import *
except:
    pass
import argparse
import gzip
import yaml
from numpy import cumsum, mean, std, zeros

# parse arguments
parser = argparse.ArgumentParser(
    prog="python summarize-learning-experiment.py",
    description="Summarize the output of an online learning experiment.")
parser.add_argument("-g", "--discount_factor", type=float, default=0.995,
    help="Discount factor to apply when evaluating online performance.")
parser.add_argument("-f", "--fold_dirs", nargs="+", required=True,
    help="List all directories that contain runs of different folds for the "
    "current data set. Results will be averaged over all folds and runs.")
parser.add_argument("-s", "--file_ext", default="txt.gz",
    help="File extension of the files in which run results are stored.")
parser.add_argument("-o", "--output_base", #required=True,
    help="Filebase for output files. Output will be stored in OUTPUT_BASE.txt"
    " (numbers) and OUTPUT_BASE_(online|offline).pdf (plots).")
args = parser.parse_args()

is_initialized = False
agg_online_ndcg = None
add_offline_ndcg = None

count_queries = 0
count_runs = 0

# for each fold and run
for fold in args.fold_dirs:
    for filename in sorted(os.listdir(fold)):
        if not filename.endswith(args.file_ext):
            continue
        if filename.startswith("_"):
            continue
        filename = os.path.join(fold, filename)
        if os.path.getsize(filename) == 0:
           continue
        if filename.endswith(".gz"):
            fh = gzip.open(filename, "r")
        else:
            fh = open(filename, "r")
        print >> sys.stderr, "Processing %s" % filename
        count_runs += 1
        # read data from output file
        data = yaml.load(fh)
        fh.close()
        if not is_initialized:
            count_queries = len(data["online_evaluation.NdcgEval"])
            agg_online_ndcg = [ [] for i in range(count_queries) ]
            agg_offline_ndcg = [ [] for i in range(count_queries) ]
            is_initialized = True
        # aggregate (and apply discounting)
        # (i is the index of the query, i.e., i=3 means performance after the
        # third query has been observed), the second index points to
        # the run id
        for i, value in enumerate(data["online_evaluation.NdcgEval"]):
            prev = 0.0
            if i > 0:
                prev = agg_online_ndcg[i-1][-1]
            # discount + cumsum
            agg_online_ndcg[i].append(prev + args.discount_factor**i * value)
        for i, value in enumerate(data["offline_test_evaluation.NdcgEval"]):
            agg_offline_ndcg[i].append(value)

print >> sys.stderr, "Computing results for up to %d queries." % count_queries
print >> sys.stderr, "Averaging over %d folds and runs." % count_runs

# output gnuplot file:
# QUERY_COUNT OFFLINE_MEAN OFFLINE_STD ONLINE_MEAN ONLINE_STD
if args.output_base:
    out_filename = "%s.txt" % args.output_base
    out_fh = open(out_filename, "w")
else:
    out_fh = sys.stdout
for i in range(count_queries):
    print >> out_fh, "%d %.5f %.5f %.5f %.5f" % (i,
    mean(agg_offline_ndcg[i]), std(agg_offline_ndcg[i]),
    mean(agg_online_ndcg[i]), std(agg_online_ndcg[i])) 
if args.output_base:
    out_fh.close()

