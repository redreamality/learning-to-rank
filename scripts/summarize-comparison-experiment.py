#!/usr/bin/python

# KH, 2012/08/31
# Summarizes the output of an online evaluation experiment. 

import argparse
import gc
import gzip
import os
import os.path
import sys
import yaml

from scipy import stats
from numpy import array, mean, sqrt, std, zeros

try:
    from include import *
except:
    pass

from lerot.utils import get_binomial_ci

# TODO: add target values (list); take list of metrics; take "every" and print
# results every so many runs

# parse arguments
parser = argparse.ArgumentParser(
    prog="python summarize-evaluation-experiment.py",
    description="Summarize the output of an online evaluation experiment.")
parser.add_argument("-e", "--eval_file", help="File in which per query / per "
    "feature evaluation scores are stored (to determine ground truth).")
parser.add_argument("-f", "--fold_dirs", nargs="+", required=True,
    help="List all directories that contain runs of different folds for the "
    "current data set. Results will be averaged over all folds and runs.")
parser.add_argument("-m", "--metrics", nargs="+", required=True,
    help="Summarize results for the given metric (e.g. 'hist_outcomes.HistBala"
    "ncedInterleave', metric names are split on '.')")
parser.add_argument("-s", "--file_ext", default="txt.gz",
    help="File extension of the files in which run results are stored.")
parser.add_argument("-o", "--output_base", #required=True,
    help="Filebase for output files. Output will be stored in OUTPUT_BASE.txt"
    " (numbers) and OUTPUT_BASE_(online|offline).pdf (plots).")
parser.add_argument("-t", "--target_values", nargs="+", type=int,
    help="A list of target value x that specifies after how many queries"
    "results should be reported (e.g., after 1, 10, and 100 queries).")
parser.add_argument("-p", "--print_every", type=int, help="Specify after how"
    " many runs results should be printed (e.g., every 20). If not set, only"
    " final results are printed.")
# doesn't work correctly
#parser.add_argument("-z", "--remove_zero_clicks", type=bool, help="If set,"
#    " remove impressions with zero clicks.")
args = parser.parse_args()

is_initialized = False
results = {}
count_queries = 0
count_runs = 0
metrics = {}
#if args.remove_zero_clicks:
#    click_counts = {}
for metric_name in args.metrics:
    metrics[metric_name]= metric_name.split(".", 1)
    #if args.remove_zero_clicks:
    #    if metric_name.startswith("hist"):
    #        click_counts[metric_name] = ["hist_click_counts",
    #            metrics[metric_name][-1]]
    #    else:
    #        click_counts[metric_name] = ["live_click_counts",
    #            metrics[metric_name][-1]]

has_eval_file = False
if hasattr(args, "eval_file") and args.eval_file:
    has_eval_file = True
    gc.disable()
    fh = open(args.eval_file, "r")
    ground_truth = yaml.load(fh)
    fh.close()
    gc.enable()

# for each fold and run
for fold in args.fold_dirs:
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
        print >> sys.stderr, "Processing %s" % filename
        count_runs += 1
        # read data from output file
        gc.disable()
        data = yaml.load(fh)
        gc.enable()
        fh.close()
        if not is_initialized:
            count_queries = len(data["query_ids"])
            for metric in metrics:
                results[metric] = {}
                results[metric]["abs"] = { i:[] for i in args.target_values }
            is_initialized = True
        # get ground truth for the current source and target rankers
        if has_eval_file:
            gt_diff = (mean(ground_truth["scores"][data["target_pair"][1]])
                - mean(ground_truth["scores"][data["target_pair"][0]]))
        else:
            gt_diff = mean(data["ndcg_diffs"])
        # aggregate results (i is the index of the query, i.e., i=3 means
        # performance after the third query has been observed), the second
        # index points to the run id
        for metric_name, metric in metrics.items():
            temp_data = data
            for metric_part in metric:
                if metric_part in temp_data:
                    temp_data = temp_data[metric_part]
                else:
                    raise ValueError("Metric %s not found in file %s."
                        % (".".join(metric), filename))
            for i in args.target_values:
                ignore = 0
                #if args.remove_zero_clicks:
                #    ignore = (i - mean(data[click_counts[metric_name][0]]
                #        [click_counts[metric_name][1]][:i]))
                outcome_i = mean(temp_data[:i+ignore])
                value = 0
                if ((outcome_i < 0 and gt_diff < 0) or
                    (outcome_i == 0 and gt_diff == 0) or
                    (outcome_i > 0 and gt_diff > 0)):
                    value = 1
                results[metric_name]["abs"][i].append(value)

        if count_runs % args.print_every == 0:
            if args.output_base:
                out_filename = "%s-%d.txt" % (args.output_base, count_runs)
                out_fh = open(out_filename, "w")
            else:
                out_fh = sys.stdout
            print >> out_fh, "# query ", ", ".join(["%s (m, lower, upper)" % metric_name
                for metric_name in args.metrics])
            print >> out_fh, "# Results for %d queries, %d folds / runs. " % (
                count_queries, count_runs)
            for i in args.target_values:
                if i > count_queries:
                    break
                print >> out_fh, "%d " % i,
                for metric_name in args.metrics:
                    p_hat = mean(results[metric_name]["abs"][i])
                    ci = get_binomial_ci(p_hat, count_runs)
                    print >> out_fh, "%g %g %g" % (
                        # mean over 0 (wrong) / 1 (correct) => binomial
                        p_hat, ci[0], ci[1]),
                print >> out_fh, ""
            if args.output_base:
                out_fh.close()
