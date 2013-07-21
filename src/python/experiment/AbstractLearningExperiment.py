import random
from utils import get_class
#from retrieval_system import AbstractOracleSystem


class AbstractLearningExperiment:

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
        self.system = self.system_class(self.feature_count, self.system_args)
        #if isinstance(self.system, AbstractOracleSystem):
        #    self.system.set_test_queries(self.test_queries)
        self.evaluations = {}
        for evaluation in args["evaluation"]:
            self.evaluation_class = get_class(evaluation)
            self.evaluations[evaluation] = self.evaluation_class()

    def _sample_qid(self, query_keys, query_count, query_length):
            if self.query_sampling_method == "random":
                return query_keys[random.randint(0, query_length - 1)]
            elif self.query_sampling_method == "fixed":
                return query_keys[query_count % query_length]

    def run(self):
        raise NotImplementedError("Derived class needs to implement run.")
