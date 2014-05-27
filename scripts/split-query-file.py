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

# KH, 2012/09/12

from include import *
import argparse
import gzip
from query import QueryStream


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(prog="python split-query-file.py",
        description="Split a data set into files per query.")
    parser.add_argument("input_file")
    parser.add_argument("output_path")
    parser.add_argument("feature_count", type=int,
        help="The number of features per document.")
    args = parser.parse_args()
    print "Reading data from %s, writing query files to %s." % (
        args.input_file, args.output_path)

    # check output dir
    if os.listdir(args.output_path):
        raise ValueError("Output path should be an empty directory: %s" %
            args.output_path)

    # initialize query stream
    if args.input_file.endswith(".gz"):
        input_fh = gzip.open(args.input_file)
    else:
        input_fh = open(args.input_file) 
    qs = QueryStream(input_fh, args.feature_count, preserve_comments=True)

    # process queries (only keeps one query in memory at a time, safes memory,
    # but not time)
    query_count = 0
    for query in qs:
        # open output file
        output_file = os.path.join(args.output_path,
            "query-%s.txt.gz" % query.get_qid())
        if os.path.exists(output_file):
            raise ValueError("File already exists: %s" % output_file)
        output_fh = gzip.open(output_file, "w")
        query.write_to(output_fh)
        query_count += 1

    print "Finished processing %d queries." % query_count
