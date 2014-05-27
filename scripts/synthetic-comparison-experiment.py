#!/usr/bin/python

# KH, 2012/09/12
# Runs a synthetic comparison experiment.

import logging
import argparse
import os
import random
import sys
import traceback
import yaml

import numpy as np
from numpy.linalg import norm

from lerot.query import load_queries
from lerot.utils import get_class

def run(run_id, experimenter, args):
    logging.info("run %d starts" % run_id)
    # initialize log file
    log_file = os.path.join(args["output_dir"], "%s-%d.json" % (
        args["output_prefix"], run_id))
    log_fh = open(log_file, "w")
    # Pass the run_id number such that we know which pair of rankers
    # we should be interleaving (model data).
    if args.get("num_random_draws") is not None:
        args["ranker_pair_idx"] = random.choice(xrange(args["num_runs"]))
    else:
        args["ranker_pair_idx"] = run_id
    try:
        # initialize experiment
        experiment = experimenter(log_fh, args)
        # run experiment
        experiment.run()
    except Exception as e:
        traceback.print_exc()
        logging.error('Error occured %s: %s' % (type(e), e))
        os.remove(log_file)

# initialize and run a learning experiment
if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        prog="python synthetic-comparison-experiment.py",
        description="""
        Construct and run a comparison experiment with synthetic data.
        Provide either the name of a config file from which the experiment
        configuration is read, or provide all arguments listed under Command
        line. If both are provided the config file is ignored.""",
        usage="%(prog)s FILE | DETAILS")

    # option 1: use a config file
    file_group = parser.add_argument_group("FILE")
    file_group.add_argument("-f", "--file", help="Filename of the config file "
        "from which the experiment details should be read.")

    # option 2: specify all experiment details as arguments
    detail_group = parser.add_argument_group("DETAILS")
    detail_group.add_argument("--verbose", type=bool, default=False,
        help="If set, print more debug information to stderr.")
    detail_group.add_argument("--num_runs", type=int,
        help="Number of runs (how many times to repeat the experiment).")
    detail_group.add_argument("--run_start_id", type=int, default=0,
        help="Starting id for numbering run files.")
    detail_group.add_argument("--processes", type=int,
        help="Number of processes if the experiments are to be run in parallel.")
    detail_group.add_argument("--num_repeat_interleaving", type=int, default=1,
        help="Number of times we repeat the interleaving procedure, "
        "compute outcomes and click metrics.")
    detail_group.add_argument("--num_random_draws",
        type=int, default=None,
        help="If set, rankers and queries (one query per ranker pair) are drawn randomly. "
        "This option only makes sense for model data.")
    detail_group.add_argument("--system_comparison", default="none",
        help="Can be 'none', 'pareto' or 'evaluation.MetricName'. "
            "This is used to identify which system is better in the pair. "
            "If this flag is set to 'pareto', "
            "we will only interleave pair of rankers when "
            "one ranker parate-dominates another "
            "(according to --pareto_um_class).")
    detail_group.add_argument("--user_model",
        help="Class implementing a user model.")
    detail_group.add_argument("--user_model_args",
        help="Arguments for initializing the user model.")
    detail_group.add_argument("--pareto_um_class",
        help="Class implementing a user model for pareto dominance "
             "(only examination probabilities are used).")
    detail_group.add_argument("--pareto_um_args",
        help="Arguments for initializing the user model for pareto dominance.")

    detail_group.add_argument("--evaluation_methods", nargs="*",
        help="List of zero or more evaluation methods to run.")
    detail_group.add_argument("--evaluation_methods_args", nargs="*",
        help="Arguments for the evaluation methods (one entry per method,"
             " in the same order).")

    detail_group.add_argument("--offline_metrics",
        type=str, default="",
        help="Comma-separated list of classes implementing offline metrics. "
             "The class has to have a get_value(self, labels, cutoff) method. "
             "The value of the corresponding offline metric will be computed "
             "for the original A and B rankings and the interleaved list L "
             "and added to the result JSON.")
    detail_group.add_argument("--compute_online_metrics",
        type=bool, default=False,
        help="If set, the online metrics will be computed.")
    detail_group.add_argument("--compute_interleaved_metrics",
        type=bool, default=False,
        help="If set, the metrics of the interleaved system will be computed "
        "and compared to the original systems A and B. The degradation is "
        "aslso computed as measured by offline or online quality metrics.")
    # The retrieval system maintains ranking functions, accepts queries and
    # generates result lists, and in return receives user clicks to learn from.
    detail_group.add_argument("-o", "--output_dir",
        help="(Empty) directory for storing output generated by this"
        " experiment. Subdirectory for different folds will be generated"
        "automatically.")
    detail_group.add_argument("--output_prefix",
        help="Prefix to be added to output filenames, e.g., the name of the "
        "data set, fold, etc. Output files will be stored as OUTPUT_DIR/"
        "PREFIX-RUN_ID.json")
    detail_group.add_argument("--output_dir_overwrite",
        type=bool, default=False,
        help="Set to true to overwrite existing output directories. False by "
        "default to prevent accidentally deleting previous results.")
    detail_group.add_argument("--experimenter", help="Experimenter class name.")

    detail_group.add_argument("--rankings",
        help="Method to generate input rankings. E.g., 'model' or 'synthetic'.")
    # Settings for model rankings.
    # These args are only looked at if '--rankings' is set to 'model'.
    detail_group.add_argument("--run_dir",
        help="Directory from which to load the runs (TREC style).")
    detail_group.add_argument("--qrel_file",
        help="File from which to load the qrels (TREC style).")
    detail_group.add_argument("--Iprob_file",
        help="File with vertical orientation (intent probabilites) values.")
    detail_group.add_argument("--vert_map_file",
        help="Mapping from the vertical names to their ids.")
    detail_group.add_argument("--ideal_page_as_rbp",
        help="Path to a file containing ideal aggregated search page (for AS_RBP)")
    # Synthetic rankings settings (only looked at if --rankings is set to 'synthetic')
    detail_group.add_argument("--result_length", type=int,
        help="Length of the result lists to show to users for each query. "
             "Excludes vertical documents when using 'fixed' vertical placement.")
    detail_group.add_argument("--num_relevant", # default="random",
        help="Number of relevant documents in the synthetic document lists.")
    detail_group.add_argument("--verticals")
    detail_group.add_argument("--vertical_posmethod")
    detail_group.add_argument("--vertical_vertrel")
    detail_group.add_argument("--vertical_blocksize", type=int)
    detail_group.add_argument("--vertical_placement")

    # run the parser
    args = vars(parser.parse_args())

    # Overwrite arguments from the config file.
    if args["file"]:
        with open(args["file"]) as config_file:
            file_args = yaml.load(config_file)
            for arg, value in file_args.iteritems():
                if arg not in args:
                    raise ValueError(
                        "Unknown argument: %s, run with -h for details." % arg)
                args[arg] = value

    #logging.basicConfig(filename=os.path.join(args["output_dir"],
    #'experiment.log'), level=logging.INFO)
    logging.basicConfig(format='%(asctime)s %(module)s: %(message)s',
                        level=logging.INFO)

    #logging.info("Arguments: %s" % args)

    # locate or create directory for the current fold
    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])
    elif not(args["output_dir_overwrite"]) and \
                                    os.listdir(args["output_dir"]):
        # make sure the output directory is empty
        raise Exception("Output dir %s is not an empty directory. "
        "Please use a different directory, or move contents out of the way." %
         args["output_dir"])
    config_bk = os.path.join(args["output_dir"], "config_bk.yml")
    logging.info("Backing up configuration to: %s" % config_bk)
    config_bk_file = open(config_bk, "w")
    yaml.dump(args, config_bk_file, default_flow_style=False)
    config_bk_file.close()

    # initialize and run the experiment num_run times
    run_start_id = args["run_start_id"]
    num_runs = args["num_runs"]
    if args.get("num_random_draws") is not None:
        # Redefine num_runs, and use args["num_runs"] only when drawing
        # pair of rankers in the run() function above.
        num_runs = args["num_random_draws"]
        assert run_start_id == 0, "Conflicting options"
    experimenter = get_class(args["experimenter"])

    # set the random seed
    random.seed(42)

    if "processes" in args and args["processes"] > 1:
        from multiprocessing import Pool
        pool = Pool(processes=args["processes"])
        for run_id in range(run_start_id, run_start_id + num_runs):
            pool.apply_async(run, (run_id, experimenter, args,))
        pool.close()
        pool.join()
    else:
        for run_id in range(run_start_id, run_start_id + num_runs):
            run(run_id, experimenter, args)
