import logging
from numpy.linalg import norm
from utils import get_cosine_similarity
from AbstractLearningExperiment import AbstractLearningExperiment


class SamplingExperiment(AbstractLearningExperiment):
    """Represents an experiment in which a retrieval system learns from
    implicit user feedback. The experiment is initialized as specified in the
    provided arguments, or config file.
    """

    def run(self):
        """Run the experiment num_runs times."""
        query_keys = sorted(self.training_queries.keys())
        query_length = len(query_keys)
        # process num_queries queries
        for query_count in range(self.num_queries):
            qid = self._sample_qid(query_keys, query_count, query_length)
            query = self.training_queries[qid]
            # get result list for the current query from the system
            result_list, i1s, i2s = self.system.get_ranked_list(query)
            # generate click feedback
            clicks = self.um.get_clicks(result_list, query.get_labels())
            # send feedback to system
            _, win = self.system.update_solution(clicks)
            if len(i1s) > 1:
                for i in range(len(i1s) - 1):
                    self.log_fh.write("%d %d %d\n" % (query_count,
                                                      i1s[i],
                                                      i2s[i]))
            self.log_fh.write("%d %d %d %d\n" % (query_count,
                                                 i1s[-1],
                                                 i2s[-1],
                                                 win))
