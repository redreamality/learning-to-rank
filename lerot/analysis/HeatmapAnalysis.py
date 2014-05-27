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

import os
import gzip
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
import numpy as np
from AbstractAnalysis import AbstractAnalysis


class HeatmapAnalysis(AbstractAnalysis):
    def __init__(self, *parms):
        AbstractAnalysis.__init__(self, *parms)
        self.analyticsfilename = os.path.join(self.analyticsroot,
                                              "heatmap.html")
        self.template = self.env.get_template('heatmap.html')
        self.summaries = {}

    def _update(self, um, data, fold, run, filename):
        if not um in self.summaries:
            self.summaries[um] = {}
        if not data in self.summaries[um]:
            self.summaries[um][data] = []

        if filename.endswith(".gz"):
            fh = gzip.open(filename, "r")
        else:
            fh = open(filename, "r")
        yamldata = yaml.load(fh, Loader=Loader)
        fh.close()
        if not yamldata or not "final_weights" in yamldata:
            return False
        weights = np.array(yamldata["final_weights"])
        weights /= np.linalg.norm(weights)
        self.summaries[um][data].append((yamldata["offline_ndcg"][-1],
                                       run,
                                       weights))

        return True

    def finish(self):
        avgs = {}
        for um in self.summaries:
            avgs[um] = {}
            for d in self.summaries[um]:
                avgs[um][d] = []
                try:
                    for i in range(len(self.summaries[um][d][0][2])):
                        avgs[um][d].append(np.var(np.array([self.summaries[um][d][j][2][i] for j in range(len(self.summaries[um][d]))])))
                except:
                    pass
        fh = open(self.analyticsfilename, 'w')
        fh.write(self.template.render(heatmaps=self.summaries,
                                      avgs=avgs))
        fh.close()
        return self.analyticsfilename


if __name__ == "__main__":
    import sys
    a = HeatmapAnalysis(sys.argv[1])
    a.update()
    print a.finish()
