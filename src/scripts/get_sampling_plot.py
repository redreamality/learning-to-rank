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
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
import argparse
import glob
import gzip
import matplotlib
matplotlib.use('Agg')
import pylab as P
import pickle
import sys
import os
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", help="directory that holds an "
                        " anaylics directory.", nargs="+")
    parser.add_argument("--um", default="per", nargs="+")
    parser.add_argument("-m", "--metric", type=str, default="binary_diff", nargs="+")
    parser.add_argument("--reload", action="store_true", default=False)
    args = parser.parse_args()
    args.metric = [args.metric]
    args.um = [args.um]
    uniqvalues = args.root_dir
    uniq = str(hash(tuple(uniqvalues)))
    uniqfile = os.path.join(basedir, 'out', uniq + ".ndcgpoints.pickle")
    if not os.path.exists(uniqfile) or args.reload:
        print "Reloading"
        ndcgpoints = {}
        exps = []
        for indir in args.root_dir:
            indir = os.path.realpath(indir)
            exp = indir.split("/")[-2]
            ndcgpoints[exp] = {}
            print exp
            exps.append(exp)
            print os.path.join(indir, "output", "*")
            for um in glob.glob(os.path.join(indir, "output", "*")):
                print "um", um
                for data in glob.glob(os.path.join(um, "*")):
                    print "data", data
                    for fold in glob.glob(os.path.join(data, "*")):
                        print "fold", fold
                        for f in glob.glob(os.path.join(fold, "*.txt.gz")):
                            parts = os.path.normpath(os.path.abspath(f)).split(os.sep)
                            umshort, datashort, foldshort = parts[-4:-1]
                            print f, umshort, datashort
                            if not umshort in args.um:
                                continue
                            if not umshort in ndcgpoints[exp]:
                                ndcgpoints[exp][umshort] = {}
                            if not datashort in ndcgpoints[exp][umshort]:
                                ndcgpoints[exp][umshort][datashort] = {}
                            if f.endswith(".gz"):
                                fh = gzip.open(f, "r")
                            else:
                                fh = open(f, "r")
                            yamldata = yaml.load(fh, Loader=Loader)
                            fh.close()
    
                            for metric in args.metric:
                                if not metric in ndcgpoints[exp][umshort][datashort]:
                                    ndcgpoints[exp][umshort][datashort][metric] = []
                                scores = yamldata["%s" % (metric)]
                                ndcgpoints[exp][umshort][datashort][metric].append(scores)

        pickle.dump(ndcgpoints, open(uniqfile, "w"))
    else:
        ndcgpoints = pickle.load(open(uniqfile, "r"))
        exps = ndcgpoints.keys()

    datas = []
    for e in ndcgpoints:
        for u in ndcgpoints[e]:
            datas += ndcgpoints[e][u].keys()

    datas = sorted(list(set(datas)))

    rawlines = [1,0]
    lines = [[1,0],[1,1],[4,2,1,2],[4,2,1,2,1,2]]
    colors = ['b','g','r','c','m','y','k']

    for data in datas:
        for user in args.um:
            cindex = 0
            for exp in exps:
                label = exp
                color = colors[cindex % len(colors)]
                cindex += 1
                lindex = 0
                if not user in ndcgpoints[exp]:
                    continue
                if not data in ndcgpoints[exp][user]:
                    continue
                for metric in ndcgpoints[exp][user][data]:
                    count = 0
                    aggregation = None
                    for rawline in ndcgpoints[exp][user][data][metric]:
                        count += 1
                        if aggregation == None:
                            aggregation = rawline
                        else:
                            for i, x in enumerate(rawline):
                                aggregation[i] += x
                    cum = [0.0]
                    for i in range(len(aggregation)):
                        aggregation[i] /= count
                        if np.isnan(aggregation[i]):
                            cum.append(cum[-1])
                        else:
                            cum.append(cum[-1] + aggregation[i])

                    l = P.plot(cum, color, label="%s-%s" % (label, metric))
                    l[0].set_dashes(lines[lindex])
            l = P.legend(loc=4)
            outfile = os.path.join(basedir, "out",
                                   "%s-%s-%s.pdf" % (uniq,
                                                     user,
                                                     data))
            P.savefig(outfile,
                      format='pdf')
            P.clf()
            P.ioff()

            print outfile
