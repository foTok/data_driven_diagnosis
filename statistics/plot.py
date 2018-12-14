import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_errorbar(mean, std, conf, xlabel, ylabel, xticklabel, legend, view=True, file=None, ylimt=None, text=True):
    '''
    Args:
        mean: 2d np.array. mean values.
        std: 2d np.array. stard error.
        conf: a float. the confidence.
        xlabel: a string.
        ylabel: a string.
        xticklabel: a tuple of strings.
        legend: a tuple of strings.
        view: if show this figure. True by default.
        file: the file path and name to save this figure. None means not save.
        ylimt: RT.
    '''
    M, N = mean.shape
    ind = np.arange(N)  # the x locations for the groups
    width = 1/(M+1)     # the width of the bars

    temp_conf = norm.ppf(1-(1-conf)/2)
    conf_interval = [[temp_conf * i for i in tmp_std] for tmp_std in std]

    _, ax = plt.subplots()
    rects = [0]*M
    for i in range(M):
        rects[i] = ax.bar(ind + width*i, mean[i], width, yerr=conf_interval[i])

    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_xticks(ind + (1-width)/2.)
    ax.set_xticklabels(xticklabel)
    if ylimt is not None:
        ax.set_ylim(ylimt[0], ylimt[1])
    ax.legend(tuple([rects[i][0] for i in range(M)]), legend, fontsize=10)

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom')
    if text:
        for _rects in rects:
            autolabel(_rects)
    if view:
        plt.show()
    if file is not None:
        plt.savefig(file)
