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
#matplotlib.use('WX')
#import numpy as N
import pylab as P
import pickle
import sys
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", help="directory that holds an "
                        " anaylics directory.", nargs="+")
    parser.add_argument("--um", default="inf", nargs="+")
    parser.add_argument("--nofail", action="store_true", default=False)
    parser.add_argument("--aggregate", default="mean", nargs="+",
                        choices=["mean", "max", "min"])
    parser.add_argument("--raw", action="store_true", default=False)
    parser.add_argument("--reload", action="store_true", default=False)
    parser.add_argument("--plotperexp", action="store_true", default=False)
    parser.add_argument("-m", "--metric", type=str, default="NdcgEval", nargs="+")
    parser.add_argument("--traintest", type=str, default="test", choices=["train", "test"])
    args = parser.parse_args()
    
    if not type(args.aggregate) is list:
        args.aggregate = [args.aggregate]
    if not type(args.metric) is list:
        args.metric = [args.metric]
    if not type(args.um) is list:
        args.um = [args.um]

    uniqvalues = args.root_dir + args.aggregate + args.um + args.metric + [args.raw, args.traintest]
    uniq = str(hash(tuple(uniqvalues)))
    uniqfile = os.path.join(basedir, 'out', uniq + ".ndcgpoints.pickle")
    if not os.path.exists(uniqfile) or args.reload:
        print "Reloading"
        for indir in args.root_dir:
            if not os.path.exists(indir):
                parser.error("root_dir %s does not exist" % indir)
            indir = os.path.realpath(indir)
            exp = indir.split("/")[-2]
            filename = os.path.join(indir, "analytics", "summary.yml")
            if not args.nofail and not os.path.exists(filename):
                parser.error("there is no summary at %s for experiment %s" %
                             (filename, exp))
        ndcgpoints = {}
        exps = []
        for indir in args.root_dir:
            indir = os.path.realpath(indir)
            exp = indir.split("/")[-2]
            ndcgpoints[exp] = {}
            print exp
            filename = os.path.join(indir, "analytics", "summary.yml")
            if os.path.exists(filename):
                fh = open(filename, "r")
                try:
                    summary = yaml.load(fh, Loader=Loader)
                except yaml.parser.ParserError:
                    if not args.nofail:
                        parser.error("There is an error in %s for experiment %s" %
                             (filename, exp))
                if summary:
                    ndcgpoints[exp] = summary
                elif not args.nofail:
                    parser.error("No summary in %s for experiment %s" %
                             (filename, exp))
                fh.close()
   
            exps.append(exp)
            if args.raw:
                print "loading RAW"
                for um in glob.glob(os.path.join(indir, "output", "*")):
                    for data in glob.glob(os.path.join(um, "*")):
                        for fold in glob.glob(os.path.join(data, "*")):
                            for f in glob.glob(os.path.join(fold, "*.txt.gz")):
                                print "loading", f
                                parts = os.path.normpath(os.path.abspath(f)).split(os.sep)
                                umshort, datashort, foldshort = parts[-4:-1]
                                if not umshort in args.um:
                                    continue
                                if not 'raw' in ndcgpoints[exp]:
                                    ndcgpoints[exp]['raw'] = {}
                                if not umshort in ndcgpoints[exp]['raw']:
                                    ndcgpoints[exp]['raw'][umshort] = {}
                                if not datashort in ndcgpoints[exp]['raw'][umshort]:
                                    ndcgpoints[exp]['raw'][umshort][datashort] = {}
                                if not foldshort in ndcgpoints[exp]['raw'][umshort][datashort]:
                                    ndcgpoints[exp]['raw'][umshort][datashort][foldshort] = {}
                                print f, umshort, datashort, foldshort
                                if f.endswith(".gz"):
                                    fh = gzip.open(f, "r")
                                else:
                                    fh = open(f, "r")
                                yamldata = yaml.load(fh, Loader=Loader)
                                fh.close()
                                #ndcgpoints[exp]['raw'][umshort][datashort][foldshort].append(yamldata["offline_ndcg"])

                                for metric in args.metric:
                                    if not metric in ndcgpoints[exp]['raw'][umshort][datashort][foldshort]:
                                        ndcgpoints[exp]['raw'][umshort][datashort][foldshort][metric] = []
                                    scores = yamldata["offline_%s_evaluation.%s" % (args.traintest, metric)]
                                    #prev = 0.0
                                    #scores = []
                                    #for s in rawscores:
                                    #    scores.append(s+prev)
                                    #    prev += s
                                    ndcgpoints[exp]['raw'][umshort][datashort][foldshort][metric].append(scores)

        pickle.dump(ndcgpoints, open(uniqfile, "w"))
    else:
        ndcgpoints = pickle.load(open(uniqfile, "r"))
        exps = ndcgpoints.keys()

    datas = []
    for e in ndcgpoints:
        for u in ndcgpoints[e]:
            datas += ndcgpoints[e][u].keys()
        if 'raw' in ndcgpoints[e]:
            for u in ndcgpoints[e]['raw']:
                datas += ndcgpoints[e]['raw'][u].keys()

    datas = sorted(list(set(datas)))

    rawlines = [1,0]
    lines = [[1,0],[1,1],[4,2,1,2],[4,2,1,2,1,2]]
    colors = ['b','g','r','c','m','y','k']

    for data in datas:
        for user in args.um:
            if not args.plotperexp:
                P.clf()
                P.ioff()
            cindex = 0
            for exp in exps:
                try:
                    label, seq = mapping[exp]
                except:
                    label = exp
                if args.plotperexp:
                    P.clf()
                    P.ioff()
                color = colors[cindex % len(colors)]
                cindex += 1
                lindex = 0
                if args.raw:
                    if not 'raw' in  ndcgpoints[exp]:
                        continue
                    if not user in ndcgpoints[exp]['raw']:
                        continue
                    if not data in ndcgpoints[exp]['raw'][user]:
                        continue
                    if not user in ndcgpoints[exp]:
                        aggregation = {}
                        for metric in args.metric:
                            aggregation[metric] = None
                        count = 0
                    for fold in ndcgpoints[exp]['raw'][user][data]:
                        for metric in ndcgpoints[exp]['raw'][user][data][fold]:
                            for rawline in ndcgpoints[exp]['raw'][user][data][fold][metric]:
                                w = rawline[:10000]
                                if not user in ndcgpoints[exp]:
                                    count += 1
                                    if aggregation[metric] == None:
                                        aggregation[metric] = w
                                    else:
                                        for i, x in enumerate(w):
                                            aggregation[metric][i] += x
                                #l = P.plot(w, color, linewidth=.3)
                                #l[0].set_dashes(rawlines)
                        if args.plotperexp:
                            l = P.legend(loc=4)
                            #P.ylim(0., 1.0)
                            P.savefig(os.path.join(basedir, "out", "%s-offline-%s-%s-%s-%s-%s.pdf" % (uniq, args.metric, user, data, fold, label)), format='pdf')
                            P.clf()
                            P.ioff()
                    if not user in ndcgpoints[exp]:
                        for metric in args.metric:
                            for i in range(len(aggregation[metric])):
                                aggregation[metric][i] /= count
                for agg in args.aggregate:
                    if not user in ndcgpoints[exp]:
                        if 'raw' in ndcgpoints[exp]:
                            for metric in args.metric:
                                w = aggregation[metric][:1000]
                                seq = lines[lindex % len(lines)]
                                #l = P.plot(w, color, label="-".join([label, agg, metric]))
                                l = P.plot(w, color, label=label)
                                l[0].set_dashes(seq)
                                P.legend()
                                lindex += 1
                            continue
                        else:
                            continue
                    else:
                        if not data in ndcgpoints[exp][user]:
                            continue
                        if not ndcgpoints[exp][user][data]:
                            continue
                        w = [get(l, "offline", agg) for l in 
                            ndcgpoints[exp][user][data][:10000]]
                    print label, user, data, agg
                    seq = lines[lindex % len(lines)]
                    l = P.plot(w, color, label=label + "-" + agg)
                    l[0].set_dashes(seq)
                    lindex += 1
                if args.plotperexp:
                    #l = P.legend(loc=4)
                    #P.ylim(0., 1.0)
                    P.savefig(os.path.join(basedir, "out", "%s-offline-%s-%s-%s-%s-%s.pdf" % (uniq, args.metric, '-'.join(args.aggregate), user, data, label)), format='pdf')

            if not args.plotperexp:
                l = P.legend(loc=4)
                P.savefig(os.path.join(basedir, "out", "%s-offline-%s-%s-%s-%s.pdf" % (uniq, args.metric, '-'.join(args.aggregate), user, data)), format='pdf')
