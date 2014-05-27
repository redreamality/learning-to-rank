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

# MZ, 2014/03/30
"""
Runs an online learning experiment. The "environment" logic is located here,
e.g., sampling queries, observing user clicks, external evaluation of result
lists
"""

from numpy.linalg import norm
from numpy import zeros, sign

from .AbstractLearningExperiment import AbstractLearningExperiment
from ..utils import get_class


class PrudentLearningExperiment(AbstractLearningExperiment):
    """Represents an experiment in which a retrieval system learns from
    implicit user feedback. The experiment is initialized as specified in the
    provided arguments, or config file.
    """
    
    def __init__(self, training_queries, test_queries, feature_count, log_fh,
                 args):
        AbstractLearningExperiment.__init__(self,training_queries,test_queries,
                                           feature_count,log_fh,args)
        self.system_class = get_class(
                            "retrieval_system.PrudentListwiseLearningSystem")
        self.system = self.system_class(self.feature_count, self.system_args)
    
    def run(self):
        """Run the experiment num_runs times."""
        query_keys = sorted(self.training_queries.keys())
        query_length = len(query_keys)
        summary = {}
        self.system.alpha = self.system.delta
        
        # process num_queries queries
        for query_count in range(self.num_queries):
            #previous_solution_w = self.system.get_solution().w
            qid = self._sample_qid(query_keys, query_count, query_length)
            query = self.training_queries[qid]
            num_comparisons = 101
            outcomes = zeros(num_comparisons)
            # Perform num_comparisons comparisons between 
            for ind in range(num_comparisons):
                # get result list for the current query from the system
                result_list = self.system.get_ranked_list(query,ind==0)
                # generate click feedback
                clicks = self.um.get_clicks(result_list, query.get_labels())
                # get outcome
                outcomes[ind] = self.system.get_outcome(clicks)
                if ind == 10:
                    score = sign(outcomes).sum()
                    if score <= 0 or score > 4:
                        break
            # If there were more wins for the candidate ranker, 
            # move (all the way) there:
            if (outcomes > 0).sum() > (outcomes < 0).sum():
                current_solution = self.system.update_solution()
                summary["weights_at_iteration_"+str(query_count)] = \
                                                current_solution.w.tolist()
            else:
                current_solution = self.system.get_solution()
            print query_count, "- positive outcomes = ", (outcomes > 0).sum(),
            print "  negative outcomes = ", (outcomes < 0).sum(),
            print "  len(weights) = %.3f" % norm(current_solution.w),
            print "  current_u = ", norm(self.system.current_u)
        
        
        summary["final_weights"] = current_solution.w.tolist()
        return summary
