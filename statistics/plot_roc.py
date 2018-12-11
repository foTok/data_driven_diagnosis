'''
The class to compute ROC and AUC
'''

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp

class plotROC:
      '''
      ROC and AUC
      '''
      def __init__(self):
            self.n_classes    = None
            self.fpr    = dict()
            self.tpr    = dict()
            self.roc_auc      = dict()

      def analyse(self, fault_num, y_label, y_score):
            '''
            Args:
                  fault_num: int, the number of faults.
                  y_label: a 1d np.array. The real labels.
                  y_score: a 2d np.array. The score.
                        batch × (fault_num+1)
            '''
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
                  plt.savefig(file)
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
