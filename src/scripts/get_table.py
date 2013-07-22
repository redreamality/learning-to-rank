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
from math import sqrt
import glob
import gzip
import numpy as N


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--measure", default="offline",
                        choices=["offline", "online"])
    parser.add_argument("-b", "--baseline", type=int, help="indicate which of the"
                        " root_dir's is the baseline", metavar="INT", default=-1)
    parser.add_argument("root_dir", nargs="+", help="directory that holds an "
                        " anaylics directory.")
    parser.add_argument("--sweep", action="store_true", default=False)
    parser.add_argument("--sort", action="store_true", default=False)
    parser.add_argument("--nofail", action="store_true", default=False)
    parser.add_argument("--aggregate", default="mean", 
                        choices=["mean", "max", "min"])
    
    
    args = parser.parse_args()
    
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
        filename = os.path.join(indir, "analytics", "summary.yml")
        if not os.path.exists(filename):
            continue
        fh = open(filename, "r")
        try:
            summary = yaml.load(fh, Loader=Loader)
        except yaml.parser.ParserError:
            if not args.nofail:
                parser.error("There is an error in %s for experiment %s" %
                     (filename, exp))
        if summary:
            exps.append(exp)
            ndcgpoints[exp] = summary
        elif not args.nofail:
            parser.error("TNo summary in %s for experiment %s" %
                     (filename, exp))
        fh.close()
        
    users = ["per", "nav", "inf"]
    #users = ["noi"]
    datas = sorted(ndcgpoints[exps[0]][users[0]].keys())
    
    if args.sweep:
        print " user & n/m & " + " & ".join([str(x) for x in [2, 5, 10, 50, 100, 200, 500]]) + "\\\\"
        for user in users:
            for n in [3, 5, 10]:
                s = "%s & %s & " % (user, n)
                l1 = []
                for m in [2, 5, 10, 50, 100, 200, 500]:
                    pat = "n%s-m%s" % (n, m)
                    #pat = "n%s" % n
                    selectedexp = [e for e in exps if e.endswith(pat)]
                    if len(selectedexp) == 0:
                        l1.append(0)
                        continue
                    exp = selectedexp[0]
    
                    try:
                        get(ndcgpoints[exp][user][datas[0]][-1], "offline", "mean")
                    except:
                        l1.append(0)
                        continue
                    l1.append(m1)
                l2 = []
                for l in l1:
                    if l == max(l1):
                        l2.append("\\textbf{%.4f}" % l)
                    elif l  == 0:
                        l2.append("-")
                    else:
                        l2.append("%.4f" % l)
    
                s += " & ".join(l2)
                s += "\\\\"
                print s
        sys.exit()
    
    if args.baseline >= 0:
        baseline = exps[args.baseline]
        exps.remove(baseline)
        if args.sort:
            exps = sorted(exps)
        exps = [baseline] + exps
    else:
        if args.sort:
            exps = sorted(exps)
    
    #users = sorted(ndcgpoints[exps[0]].keys())
    
    if args.measure == "online":
        precision = "%.2f"
    else:
        precision = "%.4f"
    
    print "\\begin{table}"
    print "\\tiny"
    print "\\begin{center}"
    print "\\begin{tabular}{ll" + "|".join(["l"] * len(exps)) + "}"
    print "\\toprule"
    print "& ", " & ".join(["\\multicolumn{1}{c}{%s}" % x
                               for x in exps]), "\\\\"
    
    
    it = 1
    for user in users:
        print "\\midrule"
        if user == "per":
            cuser = "perfect"
        elif user == "nav":
            cuser = "navigational"
        elif user == "inf":
            cuser = "informational"
        elif user == "noi":
            cuser = "noisy"
        print "\\multicolumn{" + str(len(exps) +1) + "}{c}{\emph{" + cuser + \
                " click model}} \\\\"
        print "\\midrule"
        for data in datas:
            l1 = []
            l2 = []
    
            for exp in exps:
                if not user in ndcgpoints[exp]:
                    continue
                if not data in ndcgpoints[exp][user]:
                    continue
                if not ndcgpoints[exp][user][data]:
                    continue
                m2 = get(ndcgpoints[exp][user][data][-1], args.measure, args.aggregate)
                l1.append(m2)
            for exp in exps:
                if not user in ndcgpoints[exp]:
                    l2.append("- & ")
                    continue
                if not data in ndcgpoints[exp][user]:
                    l2.append("- & ")
                    continue
                if not ndcgpoints[exp][user][data]:
                    l2.append("- & ")
                    continue
    
                m2 = get(ndcgpoints[exp][user][data][-1], args.measure, args.aggregate)
                std2 = get(ndcgpoints[exp][user][data][-1], args.measure, "std")
                significance = ""
                if args.baseline != None and baseline != exp:
                    if baseline in ndcgpoints and \
                            user in ndcgpoints[baseline] and \
                            data in ndcgpoints[baseline][user]:
                        m1 = get(ndcgpoints[baseline][user][data][-1],
                                 args.measure, args.aggregate)
                        std1 = get(ndcgpoints[baseline][user][data][-1],
                                   args.measure, "std")
                        significance = get_significance(m1, m2, std1, std2, 125)
                if m2 == max(l1):
                    l2.append("\\textbf{" + (precision % m2) + "} (%.2f) %s " %
                              (std2, significance))
                else:
                    l2.append((precision % m2) + " (%.2f) %s " % (std2, significance))
    
            #print it, "&", data, "&", " & ".join(l2), "\\\\"
            print data, "&", " & ".join(l2), "\\\\"
            it += 1
    print "\\bottomrule"
    print "\\end{tabular}"
    print "\\end{center}"
    if args.measure == "online":
        caption = """Online performance (in terms of discounted cumulative NDCG)
        when learning with interleaved comparison methods. Best runs per row are
        highlighted in bold. """
        if args.baseline:
            caption += """Statistically significant improvements (losses) from
        the %s method are indicated by \enkelop ($p=0.05$) and \dubbelop
        ($p=0.01$) (\enkelneer\ and \dubbelneer).""" % baseline
    else:
        caption = """Offline performance (in terms of NDCG) when learning with
        interleaved comparison methods.  Best runs per row are highlighted in
        bold. """
        if args.baseline != None:
            caption += """Statistically significant improvements (losses) from
            the %s method are indicated by \enkelop ($p=0.05$) and \dubbelop
            ($p=0.01$) (\enkelneer\ and \dubbelneer).""" % baseline
    print "\\caption{" + caption + "}"
    print "\\label{tab:results-" + args.measure + "}"
    print "\\end{table}"
