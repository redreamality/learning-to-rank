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

from setuptools import setup
import os.path

# Get __version__ from the source directory
dist_dir = os.path.dirname(os.path.abspath(__file__))
execfile(os.path.join(dist_dir, 'lerot/_version.py'))

setup(
    name = "Lerot",
    version = __version__,
    author = "Katja Hofmann, Anne Schuth",
    author_email = "katja.hofmann@microsoft.com, anne.schuth@uva.nl",
    description = ("This project is designed to run experiments on online\
                    learning to rank methods for information retrieval."),
    keywords = "online learning to rank for information retrieval",
    url = "https://bitbucket.org/ilps/lerot",
    download_url = "https://bitbucket.org/ilps/lerot/downloads",
    packages=(['lerot']
              + [('lerot.%s' % sub)
                 for sub in ('analysis', 'comparison', 'environment',
                             'evaluation', 'experiment', 'ranker',
                             'ranker.model', 'retrieval_system')]),
    long_description=open('README.rst', 'r').read(),
    license = "GNU Lesser General Public License",
    scripts=['scripts/learning-experiment.py',
             'scripts/meta-experiment.py',
             'scripts/summarize-learning-experiment.py']
)
