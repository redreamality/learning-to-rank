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

# KH, 2012/08/05
"""
Compute offline performance (in ndcg, letor ndcg) for given rankers and test
queries.
"""

from include import *
import argparse
import gzip
import yaml
import numpy as np
from query import load_queries
from utils import get_class

# for each fold and run of each experiment:
#    read all output files (final_weights)
#    apply weights to test queries and compute ndcg / letor ndcg
# print mean, ste over folds and runs per experiment

# parse arguments
parser = argparse.ArgumentParser(description="""
    Summarize the output of an online learning experiment.""")
parser.add_argument("-d", "--test_dir", required=True,
    help="Directory that contains the test queries (per fold).")
parser.add_argument("-t", "--test_file", default="test.txt.gz",
    help="In each fold of the test directory, the name of the test file.")
parser.add_argument("-f", "--feature_count", required=True, type=int,
    help="Number of features (has to match test queries and weight files).")
parser.add_argument("-e", "--experiment_dirs", nargs="+", required=True,
    help="List of directories that contain experiments (one per experiment). "
    "Results per experiment will be averaged over all folds and runs.")
parser.add_argument("-s", "--file_ext", default="txt.gz",
    help="File extension of the files in which run results are stored.")
args = parser.parse_args()

cutoffs = [1, 3, 10, -1]
metrics = []
scores = {}
for metric in  "evaluation.NdcgEval", "evaluation.LetorNdcgEval":
    eval_class = get_class(metric)
    eval_metric = eval_class()
    metrics.append(eval_metric)
    scores[eval_metric.__class__.__name__] = {}
    for cutoff in cutoffs:
        scores[eval_metric.__class__.__name__][cutoff] = []

# load all queries
test_queries = {}
for fold in range(1, 6):
    test_file = "".join((args.test_dir, str(fold)))
    test_file = os.path.join(test_file, args.test_file)
    qs = load_queries(test_file, args.feature_count)
    test_queries[fold] = qs

# process all experiments for all metrics
count_experiments = 0
for experiment in args.experiment_dirs:
    print "%% %s" % experiment
    count_runs = 0
    count_experiments += 1
    # process all folds and run files
    for fold_id in sorted(os.listdir(experiment)):
        fold = os.path.join(experiment, fold_id)
        fold_id = int(fold_id)
        if not os.path.isdir(fold):
            continue
        for filename in sorted(os.listdir(fold)):
            if not filename.endswith(args.file_ext):
                continue
            filename = os.path.join(fold, filename)
            if os.path.getsize(filename) == 0:
                continue
            if filename.endswith(".gz"):
                fh = gzip.open(filename, "r")
            else:
                fh = open(filename, "r")
            count_runs += 1
            # read data from output file
            data = yaml.load(fh)
            fh.close()
            weights = np.array(data["final_weights"])
            for metric in metrics:
                for cutoff in cutoffs:
                    score = metric.evaluate_all(weights,
                        test_queries[fold_id], cutoff=cutoff)
                    scores[metric.__class__.__name__][cutoff].append(score)

    # print id, mean, lower-ste, higher-ste, min, max, n, q
    for metric in metrics:
        for cutoff in cutoffs:
            print "%% %s %d" % (metric.__class__.__name__, cutoff)
            print " & %.3f $\pm$ %.3f %% %d" % (
                np.mean(scores[metric.__class__.__name__][cutoff]),
                np.std(scores[metric.__class__.__name__][cutoff]) /
                np.sqrt(count_runs), count_runs)
