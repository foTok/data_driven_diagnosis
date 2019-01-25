'''
The class to compute ROC and AUC
'''
import numpy as np
import matplotlib.pyplot as plt
import argparse
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from scipy.stats import norm

class plotROC:
      '''
      ROC and AUC
      '''
      def __init__(self):
            self.n_classes    = None
            self.fpr    = dict()
            self.tpr    = dict()
            self.roc_auc      = dict()

      def analyse(self, y_label, y_score):
            '''
            Args:
                  y_label: a 1d np.array. The real labels.
                  y_score: a 2d np.array. The score.
                        batch × (fault_num+1)
            '''
            fault_num = int(np.max(y_label))
            self.n_classes    = fault_num + 1   # 0 for normal
            y_test     = label_binarize(y_label, classes=[i for i in range(self.n_classes)])
            for i in range(self.n_classes):
                  self.fpr[i], self.tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                  self.roc_auc[i] = auc(self.fpr[i], self.tpr[i])
            # Compute micro-average ROC curve and ROC area
            self.fpr["micro"], self.tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
            self.roc_auc["micro"] = auc(self.fpr["micro"], self.tpr["micro"])

            # Compute macro-average ROC curve and ROC area
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([self.fpr[i] for i in range(self.n_classes)]))
            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(self.n_classes):
                  mean_tpr += interp(all_fpr, self.fpr[i], self.tpr[i])
            # Finally average it and compute AUC
            mean_tpr /= self.n_classes

            self.fpr["macro"] = all_fpr
            self.tpr["macro"] = mean_tpr
            self.roc_auc["macro"] = auc(self.fpr["macro"], self.tpr["macro"])

      def plot(self, key, view=True, file=None):
            if isinstance(key, int):
                  assert key < self.n_classes
            elif isinstance(key, str):
                  assert key == 'micro' or key == 'macro'
            else:
                  raise RuntimeError('Unknown type.')
            label = '{0} ROC curve (area = {1:0.4f})'.format(key, self.roc_auc[key])
            lw = 2
            plt.figure()
            plt.plot(self.fpr[key], self.tpr[key], label=label, color='deeppink', linestyle=':', linewidth=4)

            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            if file is not None:
                  plt.savefig(file+'.svg', format='svg')
            if view:
                  plt.show()
            plt.close()

      def auc(self, key):
            if isinstance(key, int):
                  assert key < self.n_classes
            elif isinstance(key, str):
                  assert key == 'micro' or key == 'macro'
            else:
                  raise RuntimeError('Unknown type.')
            return self.roc_auc[key]


def plotErrorbar(mean, std, conf, xlabel, ylabel, xticklabel, legend, view=True, file=None, ylimt=None, text=True):
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
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height+0.02,
                    '%.4f' % height,
                    ha='center', va='bottom')
    if text:
        for _rects in rects:
            autolabel(_rects)
    if view:
        plt.show()
    if file is not None:
        plt.savefig(file)


if __name__ == "__main__":
      # BPSK diagnosis time CNN
      mean = np.array([[0.3220, 0.1029]])
      std  = np.array([[0.0123, 0.0018]])
      conf = 0.95
      xlabel = 'Model'
      ylabel = 'Diagnosis Time /s'
      xticklabel = ['Cumbersome Model', 'Student Model']
      legend = ['Diagnosis Time']
      plotErrorbar(mean, std, conf, xlabel, ylabel, xticklabel, legend, text=True, ylimt=(0, 0.40))
      # BPSK diagnosis time LSTM
      mean = np.array([[1.4540, 0.3814]])
      std  = np.array([[0.0232, 0.0152]])
      conf = 0.95
      xlabel = 'Model'
      ylabel = 'Diagnosis Time /s'
      xticklabel = ['Cumbersome Model', 'Student Model']
      legend = ['Diagnosis Time']
      plotErrorbar(mean, std, conf, xlabel, ylabel, xticklabel, legend, text=True, ylimt=(0, 1.7))

      # BPSK AUC Comparison of BNs with Different Features in BPSK System
      mean = np.array([[0.65, 0.65, 0.978883333, 0.78864375]])
      std  = np.array([[0, 0, 0.003864611, 0.015253609]])
      conf = 0.95
      xlabel = 'Feature Type'
      ylabel = 'AUC'
      xticklabel = ['Original Variables', 'PCA Features', 'CNN Features', 'LSTM Features']
      legend = ['AUC']
      plotErrorbar(mean, std, conf, xlabel, ylabel, xticklabel, legend, text=True, ylimt=(0, 1.2))

      # MT AUC Comparison of Cumbersome and Student Models
      mean = np.array([[0.9984, 0.8836], [0.9971, 0.9438]])
      std  = np.array([[0.0009, 0.0264], [0.0016, 0.0221]])
      conf = 0.95
      xlabel = 'Model'
      ylabel = 'AUC'
      xticklabel = ['Cumbersome Model', 'Student Model']
      legend = ['Cumbersome Model', 'Student Model']
      plotErrorbar(mean, std, conf, xlabel, ylabel, xticklabel, legend, text=True, ylimt=(0, 1.3))

      # MT AUC Comparison of BNs with Different Features in MT System
      mean = np.array([[0.8430, 0.8258, 0.9420, 0.9731],
                       [0.8060, 0.7816, 0.8733, 0.9036]])
      std  = np.array([[0.0190, 0.0025, 0.0455, 0.0108],
                       [0.0008, 0.0006, 0.0305, 0.0065]])
      conf = 0.95
      xlabel = 'Feature Type'
      ylabel = 'AUC'
      xticklabel = ['Original Variables', 'PCA Features', 'CNN Features', 'LSTM Features']
      legend = ['Set1', 'Set2']
      plotErrorbar(mean, std, conf, xlabel, ylabel, xticklabel, legend, text=True, ylimt=(0, 1.3))


