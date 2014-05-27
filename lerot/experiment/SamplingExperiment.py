import logging
from numpy.linalg import norm
from utils import get_cosine_similarity, get_class
from AbstractLearningExperiment import AbstractLearningExperiment
import re


class SamplingExperiment(AbstractLearningExperiment):
    """Represents an experiment in which a retrieval system learns from
    implicit user feedback. The experiment is initialized as specified in the
    provided arguments, or config file.
    """
    def __init__(self, training_queries, test_queries, feature_count, log_fh,
                 args):
        """Initialize an experiment using the provided arguments."""
        self.log_fh = log_fh
        self.training_queries = training_queries
        self.test_queries = test_queries
        self.feature_count = feature_count
        # construct system according to provided arguments
        self.num_queries = args["num_queries"]
        self.query_sampling_method = args["query_sampling_method"]
        self.um_class = get_class(args["user_model"])
        self.um_args = args["user_model_args"]
        self.um = self.um_class(self.um_args)
        self.system_class = get_class(args["system"])
        self.system_args = args["system_args"]
        self.evaluation_class = get_class(args["evaluation"])
        self.evaluation = self.evaluation_class()
        self.run_count, = re.findall(r'-(\d+)\.txt',log_fh.name)
        self.system = self.system_class(self.feature_count,
                                        self.system_args, self.run_count)


    def run(self):
        """Run the experiment num_runs times."""
        query_keys = sorted(self.training_queries.keys())
        query_length = len(query_keys)
        out_str = ""
        # process num_queries queries
        for query_count in range(self.num_queries):
            qid = self._sample_qid(query_keys, query_count, query_length)
            query = self.training_queries[qid]
            # get result list for the current query from the system
            result_list, i1s, i2s = self.system.get_ranked_list(query)
            # generate click feedback
            clicks = self.um.get_clicks(result_list, query.get_labels())
            # send feedback to system
            win = self.system.update_solution(clicks)
            out_str = out_str + "%d %d %d %d\n" % (query_count,
                                                 i1s[-1],
                                                 i2s[-1],
                                                 win)
            if query_count % 10000 == 9999 or query_count == self.num_queries-1:
                self.log_fh.write(out_str)
                out_str = ""
        
        return []
