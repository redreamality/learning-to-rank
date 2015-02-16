'''
Created on 15 jan. 2015

@author: Jos
'''
from datetime import datetime
import os

import matplotlib.pyplot as plt
import numpy as np

params = {
#'text.latex.preamble': r"\usepackage{lmodern}",
#'text.usetex' : True,
#'font.size' : 11,
#'font.family' : 'lmodern',
#'text.latex.unicode': True,
}
plt.rcParams.update(params)
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#Options
params = {
          'text.usetex' : True,
          'font.size' : 11,
          'font.family' : 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params) 


from pylab import arange,pi,sin,cos,sqrt

fig_width_pt = 480.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean = (sqrt(5)-1.0)/1.5         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height =fig_width*golden_mean       # height in inches
fig_height = 240*inches_per_pt
fig_size = [fig_width,fig_height]
params = {'backend': 'ps',
     'axes.labelsize': 20,
     'text.fontsize': 20,
     'legend.fontsize': 20,
     'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': True,
    'figure.figsize': fig_size,
    'axes.facecolor': "white",
    }
plt.rcParams.update(params) 


EXP='sensitivity'
#EXP='bias'

PATH_DATA = '/Users/aschuth/Documents/lerot/lerot-PM/sigir2015short/' + EXP
PATH_PLOTS = '/Users/aschuth/Documents/lerot/lerot-PM/sigir2015short/plots/' + EXP
METHODS = ['informational', 'navigational', 'perfect']
#METHODS = ['perfect']
#METHODS = ['random']
MEASURES = ['PM', 'TDM', 'PI', 'SPM10', 'SPM100', 'SPM1000']


def evaluate():
    output = readData()

    averages = [[[np.average([np.average([output[method][fold][run][k][measure]
                                          for fold in range(5)])
                              for run in range(25)])
                  for k in range(1000)]
                 for measure in range(1, 7)]
                for method in range(len(METHODS))]
    for i, average in enumerate(averages):
        method = METHODS[i]
        std = [[np.std(np.array([output[METHODS.index(method)][fold][run][k][measure]
                                for fold in range(5)
                                for run in range(25)]))
               for k in range(1000)]
               for measure in range(1, 7)]
        visualizeError(average, MEASURES, std, imageName=method, show=False,
                       x_range=1000)


def get_files(path):
    files = []
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)):
            files.append(os.path.join(path, i))
        else:
            files += get_files(os.path.join(path, i))
    return files


def readData(path=PATH_DATA, methods=METHODS):
    '''
    OUTPUT:
    - list containing a list for each method
         containing a list for each fold
            containing a list for each run
                containing a list of iteration, probablistic_multileave,
                teamdraft_multi, probabilistic_non_bin_multi, probabilistic_inter
    '''
    output = []
    
    allfiles = get_files(path)
    
    for m in methods:
        
        files = [f for f in allfiles if m in f and "out.txt" in f]
        
        output_method = []
        print m, files
        for f in files:
            print(f)
            with open(f, "r") as myfile:
                output_file = []
                output_run = []
                for line in myfile.readlines():
                    if "RUN" in line:
                        output_run = []
                    elif line in ['\n', '\r\n']:
                        output_file.append(output_run)
                    elif 'probabilistic' in line:
                        pass
                    else:
                        output_run.append([float(l) for l in line.split()])
            output_method.append(output_file)
        output.append(output_method)
    return output

def get_significance(mean_1, mean_2, std_1, std_2, n):
   significance = "-"
   ste_1 = std_1 / np.sqrt(n)
   ste_2 = std_2 / np.sqrt(n)
   t = (mean_1 - mean_2) / np.sqrt(ste_1 ** 2 + ste_2 ** 2)
   if mean_1 > mean_2:
       # treatment is worse than baseline
       # values used are for 120 degrees of freedom
       # (http://changingminds.org/explanations/research/analysis/
       # t-test_table.htm)
       if abs(t) >= 2.62:
           significance = "\dubbelneer"
       elif abs(t) >= 1.98:
           significance = "\enkelneer"
   else:
       if abs(t) >= 2.62:
           significance = "\dubbelop"
       elif abs(t) >= 1.98:
           significance = "\enkelop"
   return significance

def visualizeError(errors, labels, std, path_plots=PATH_PLOTS, imageName='',
                   show=True, x_range=None, y_range=None):
    '''
    Show and save a graph of the errors over time

    ARGS:
    - errors: list of list of errors: for each method an list of the errors
      over time
    - labels: list of names for the methods
    - path_plots = where to save the data. If None, it wont be saved
    - imageName = name of the image if it is saved in the path_plots
    '''
    fig = plt.figure(facecolor="white")
    fig.patch.set_facecolor('white')
    
    plt.hold(True)
    colors = [('red', '-'), ('green', '-'), ('blue', '-'), ('orange', '--'), ('orange', '-.'), ('orange', ':')]
    
    vals = {}
    
    for e, s, l, c in zip(errors, std, labels, colors):
        if x_range is not None:
            e = e[:x_range]
            s = s[:x_range]
        x = np.arange(len(e))
        e = np.array(e)

        n = 50
        #plt.errorbar(x[::n], e[x[::n]], yerr=np.array(s)[x[::n]] / 3, ecolor=c[0], fmt='none')
        plt.plot(x, e, label=l, color=c[0], ls=c[1])
        
        vals[l] = (e[500], s[500])
        
        
        #plt.errorbar(x, e, yerr=np.array(s)[x[::n]] / 3, label=l, color=c[0], ls=c[1])

    print imageName

    for l in labels:
        e, s = vals[l]
        sig1 = ""
        sig2 = ""
        if "PM" in l:
            sig1 = get_significance(e, vals["PI"][0], s, vals["PI"][1], 125*2)
            sig2 = get_significance(e, vals["TDM"][0], s, vals["TDM"][1], 125*2)
        print "%s %.3f \small{(%.2f)} %s %s\t &" % (l, e, s, sig1, sig2)

    ax = plt.subplot(111)
    ax.patch.set_facecolor('white')
    #ax.spines["top"].set_visible(False)
    #ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.spines['bottom'].set_color('k')
    ax.spines['top'].set_color('k') 
    ax.spines['right'].set_color('k')
    ax.spines['left'].set_color('k')
    ax.text(30, .7, "\emph{%s}" % imageName, size=20) 
    #ax.set_yscale('log')
    plt.xlim([0, x_range])
    plt.ylim([0, .8])
    ax.tick_params(axis='both', which='major', color="k")

    plt.ylabel('$E_{bin}$', color="k")
    if imageName.lower() in ["perfect", "random"] :
        plt.legend(ncol=2, frameon=False, loc=7)
    if imageName.lower() in ["informational", "random"]:
        plt.xlabel('query impressions', color="k")

    if show:
        plt.show()
    plt.hold(False)

    if path_plots is not None:
        now = datetime.now()
        imageName = 'plot_' + imageName #+ '_'.join([str(now.hour),
                                        #            str(now.minute),
                                        #str(now.second)])
        fig.tight_layout()                               
        fig.savefig(path_plots + imageName + '.pdf', format='pdf', #transparant=True, 
                    facecolor="white")
if __name__ == '__main__':
    evaluate()
