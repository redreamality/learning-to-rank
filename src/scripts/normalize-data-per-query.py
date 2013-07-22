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

# KH, 2012/07/18

from include import *
import argparse
import gzip
import numpy as np
from query import QueryStream


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(prog="python normalize-data.py",
        description="Normalize a data set per query.")
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("feature_count", type=int,
        help="The number of features per document.")
    args = parser.parse_args()
    print "Reading data from %s, writing normalized data to %s." % (
        args.input_file, args.output_file)

    # initialize query stream
    if args.input_file.endswith(".gz"):
        input_fh = gzip.open(args.input_file)
    else:
        input_fh = open(args.input_file) 
    qs = QueryStream(input_fh, args.feature_count, preserve_comments=True)

    # open output file
    if os.path.exists(args.output_file):
        raise ValueError("Target file already exists: %s" % args.output_file)
    if args.output_file.endswith(".gz"):
        output_fh = gzip.open(args.output_file, "w")
    else:
        output_fh = open(args.output_file, "w")

    # process queries (only keeps one query in memory at a time, safes memory,
    # but not time)
    query_count = 0
    for query in qs:
        # find min and max values per feature
        feature_vectors = query.get_feature_vectors()
        min_per_feature = np.zeros(args.feature_count)
        max_per_feature = np.zeros(args.feature_count)
        for feature in range(args.feature_count):
            min_per_feature[feature] = np.min(feature_vectors[:,feature])
            max_per_feature[feature] = np.max(feature_vectors[:,feature])
        for document in range(query.get_document_count()):
            for feature in range(args.feature_count):
                if min_per_feature[feature] == max_per_feature[feature]:
                    feature_vectors[document][feature] = .0
                else:
                    feature_vectors[document][feature] = ((
                        feature_vectors[document][feature] -
                        min_per_feature[feature]) /
                        (max_per_feature[feature] - min_per_feature[feature]))
        query.write_to(output_fh)
        query_count += 1

    print "Finished processing %d queries." % query_count