#!/usr/bin/env python

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

from include import *
import argparse
from utils import get_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Analysis""",
                                     usage="%(prog)s ")
    parser.add_argument('--basedir', required=True)
    parser.add_argument('--analysis', nargs="+", required=True)
    args = parser.parse_known_args()[0]
    
    for analyse in args.analysis:
        aclass = get_class(analyse)
        a = aclass(args.basedir)
        a.update()
        print a.finish()
