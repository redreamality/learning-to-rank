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
import glob
from jinja2 import FileSystemLoader
from jinja2.environment import Environment


class AbstractAnalysis(object):
    def __init__(self, rootdir):
        self.outputdir = os.path.join(rootdir, "output")
        self.analyticsroot = os.path.join(rootdir, "analytics")
        if not os.path.exists(self.analyticsroot):
            os.makedirs(self.analyticsroot)
        self.analyticsfilename = os.path.join(self.analyticsroot,
                                             "index.html")
        self.done = []

        self.env = Environment()
        self.env.loader = FileSystemLoader(os.path.join(
                                                os.path.dirname(__file__),
                                                "templates"))

    def update_file(self, f):
        parts = os.path.normpath(os.path.abspath(f)).split(os.sep)
        um, data, fold = parts[-4:-1]
        if not f in self.done:
            if self._update(um,
                         data,
                         fold,
                         os.path.basename(f).split(".")[0],
                         f):
                self.done.append(f)

    def update(self):
        for um in glob.glob(os.path.join(self.outputdir, "*")):
            for data in glob.glob(os.path.join(um, "*")):
                for fold in glob.glob(os.path.join(data, "*")):
                    for f in glob.glob(os.path.join(fold, "*.txt.gz")):
                        self.update_file(f)
        return self.finish()

    def finish(self):
        return self.analyticsfilename

