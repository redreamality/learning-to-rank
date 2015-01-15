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


def evaluate():
    output = readData()

    errors = [[l[4] for l in output[0][0][i]] for i in range(5)]
    visualizeError(errors, METHODS)


def readData(path=PATH_DATA, methods=METHODS):
    '''
    OUTPUT:
    - list containing a list for each method
         containing a list for each fold
            containing a list for each run
                containing a list of iteration, probablistic_multileave,
                teamdraft_multi, probabilistic_non_bin_multi, probabilistic_inter
    '''
    files = []
    output = []
    for m in methods:
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


def visualizeError(errors, labels, path_plots=PATH_PLOTS, imageName=None):
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

    for e, l in zip(errors, labels):
        smoothed = ndimage.filters.gaussian_filter(e, smoothing_factor)
        plt.plot(np.arange(len(e)), smoothed, label=l)

    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.xlabel('Queries')
    plt.ylabel('Binary error')
    plt.legend()

    plt.show()
    plt.hold(False)

    if path_plots is not None:
        if imageName is None:
            now = datetime.now()
            imageName = 'plot_' + '_'.join([str(now.hour), str(now.minute),
                                           str(now.second)])
        fig.savefig(path_plots + imageName)
if __name__ == '__main__':
    evaluate()
