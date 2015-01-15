'''
Created on 15 jan. 2015

@author: Jos
'''
from datetime import datetime
import os

from scipy import ndimage

import matplotlib.pyplot as plt
import numpy as np


PATH_DATA = '../../../results/'
PATH_PLOTS = '../../../results/plots/'
METHODS = ['informational', 'navigational', 'perfect']
MEASURES = ['Probabilistic multileaving', 'Teamdraft multileaving',
            'Probabilistic Interleaving']


def evaluate():
    output = readData()

    averages = [[[np.average([np.average([output[method][fold][run][k][measure]
                                          for fold in range(5)])
                              for run in range(5)])
                  for k in range(5000)]
                 for measure in [1, 2, 4]]
                for method in range(len(METHODS))]
    for i, average in enumerate(averages):
        method = METHODS[i]
        std = [[np.std(np.array([output[METHODS.index(method)][i][j][k][measure]
                                for i in range(5)
                                for j in range(5)]))
               for k in range(5000)]
               for measure in [1, 2, 4]]
        visualizeError(average, MEASURES, std, imageName=method, show=False,
                       x_range=1000)


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
    for m in methods:
        files = []
        for i in os.listdir(path):
            if os.path.isfile(os.path.join(path, i)) and m in i:
                files.append(i)
        output_method = []
        for f in files:
            print(f)
            with open(path + f, "r") as myfile:
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
    smoothing_factor = 20
    fig = plt.figure()
    plt.hold(True)
    colors = ['blue', 'green', 'red']
    for e, s, l, c in zip(errors, std, labels, colors):
        if x_range is not None:
            e = e[:x_range]
            s = s[:x_range]
        x = np.arange(len(e))
        e = np.array(e)

#         smoothed = ndimage.filters.gaussian_filter(e, smoothing_factor)
        n = 50
        plt.errorbar(x[::n], e[x[::n]], yerr=np.array(s)[x[::n]] / 3, color=c,
                     fmt='none')
        plt.plot(x, e, label=l, color=c)

    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.xlim([0, x_range])

    plt.xlabel('Queries')
    plt.ylabel('Binary error')
    plt.legend()

    if show:
        plt.show()
    plt.hold(False)

    if path_plots is not None:
        now = datetime.now()
        imageName = 'plot_' + imageName + '_'.join([str(now.hour),
                                                    str(now.minute),
                                       str(now.second)])
        fig.savefig(path_plots + imageName + '.PDF', format='pdf')
if __name__ == '__main__':
    evaluate()
