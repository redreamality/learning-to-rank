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

from GenericExperiment import GenericExperiment
from VerticalEvaluationExperiment import VerticalEvaluationExperiment
from LearningExperiment import LearningExperiment
from PrudentLearningExperiment import PrudentLearningExperiment
from SamplingExperiment import SamplingExperiment
from MetaExperiment import MetaExperiment
from HistoricalComparisonExperiment import HistoricalComparisonExperiment
from SyntheticComparisonExperiment import SyntheticComparisonExperiment
from VASyntheticComparisonExperiment import VASyntheticComparisonExperiment


__all__ = ['GenericExperiment', 'LearningExperiment', 'MetaExperiment',
           'PrudentLearningExperiment', 'HistoricalComparisonExperiment',
           'SyntheticComparisonExperiment', 'VASyntheticComparisonExperiment']
