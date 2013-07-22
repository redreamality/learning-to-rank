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
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "Lerot",
    version = "1.0",
    author = "Katja Hofmann, Anne Schuth",
    author_email = "katja.hofmann@microsoft.com, anne.schuth@uva.nl",
    description = ("This project is designed to run experiments on online\
                    learning to rank methods for information retrieval."),
    keywords = "online learning to rank for information retrieval",
    url = "https://bitbucket.org/ilps/lerot",
    download_url = "https://bitbucket.org/ilps/lerot/downloads",
    package_dir = {'': 'src/python'},
    packages=['analysis', 'comparison', 'environment', 'evaluation',
              'experiment', 'query', 'ranker', 'retrieval_system', 'utils'],
    long_description=read('README.rst'),
    license = "GNU Lesser General Public License",
    scripts=['src/scripts/learning-experiment.py', 'src/scripts/meta-experiment.py']

)
