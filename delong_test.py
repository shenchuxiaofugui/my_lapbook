import numpy as np

import scipy.stats as st
import pandas as pd


class DelongTest():
    def __init__(self, preds1, preds2, label, threshold=0.05):
        '''
        preds1:the output of model1
        preds2:the output of model2
        label :the actual label
        '''
        self._preds1 = preds1
        self._preds2 = preds2
        self._label = label
        self.threshold = threshold
        self._show_result()

    def _auc(self, X, Y) -> float:
        return 1 / (len(X) * len(Y)) * sum([self._kernel(x, y) for x in X for y in Y])

    def _kernel(self, X, Y) -> float:
        '''
        Mann-Whitney statistic
        '''
        return .5 if Y == X else int(Y < X)

    def _structural_components(self, X, Y) -> list:
        V10 = [1 / len(Y) * sum([self._kernel(x, y) for y in Y]) for x in X]
        V01 = [1 / len(X) * sum([self._kernel(x, y) for x in X]) for y in Y]
        return V10, V01

    def _get_S_entry(self, V_A, V_B, auc_A, auc_B) -> float:
        return 1 / (len(V_A) - 1) * sum([(a - auc_A) * (b - auc_B) for a, b in zip(V_A, V_B)])

    def _z_score(self, var_A, var_B, covar_AB, auc_A, auc_B):
        return (auc_A - auc_B) / ((var_A + var_B - 2 * covar_AB) ** (.5) + 1e-8)

    def _group_preds_by_label(self, preds, actual) -> list:
        X = [p for (p, a) in zip(preds, actual) if a]
        Y = [p for (p, a) in zip(preds, actual) if not a]
        return X, Y

    def _compute_z_p(self):
        X_A, Y_A = self._group_preds_by_label(self._preds1, self._label)
        X_B, Y_B = self._group_preds_by_label(self._preds2, self._label)

        V_A10, V_A01 = self._structural_components(X_A, Y_A)
        V_B10, V_B01 = self._structural_components(X_B, Y_B)

        auc_A = self._auc(X_A, Y_A)
        auc_B = self._auc(X_B, Y_B)

        # Compute entries of covariance matrix S (covar_AB = covar_BA)
        var_A = (self._get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1 / len(V_A10) + self._get_S_entry(V_A01, V_A01, auc_A,
                                                                                                    auc_A) * 1 / len(
            V_A01))
        var_B = (self._get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1 / len(V_B10) + self._get_S_entry(V_B01, V_B01, auc_B,
                                                                                                    auc_B) * 1 / len(
            V_B01))
        covar_AB = (self._get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1 / len(V_A10) + self._get_S_entry(V_A01, V_B01,
                                                                                                       auc_A,
                                                                                                       auc_B) * 1 / len(
            V_A01))

        # Two tailed test
        z = self._z_score(var_A, var_B, covar_AB, auc_A, auc_B)
        p = st.norm.sf(abs(z)) * 2

        return z, p

    def _show_result(self):
        z, p = self._compute_z_p()
        print(f"z score = {z:.5f};\np value = {p:.5f};")
        if p < self.threshold:
            print("There is a significant difference")
        else:
            print("There is NO significant difference")

#!/usr/bin/env python3

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy.stats as st
from scipy import stats



# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
   """Computes midranks.
   Args:
      x - a 1D numpy array
   Returns:
      array of midranks
   """
   J = np.argsort(x)
   Z = x[J]
   N = len(x)
   T = np.zeros(N, dtype=np.float)
   i = 0
   while i < N:
       j = i
       while j < N and Z[j] == Z[i]:
           j += 1
       T[i:j] = 0.5*(i + j - 1)
       i = j
   T2 = np.empty(N, dtype=np.float)
   # Note(kazeevn) +1 is due to Python using 0-based indexing
   # instead of 1-based in the AUC formula in the paper
   T2[J] = T + 1
   return T2


def compute_midrank_weight(x, sample_weight):
   """Computes midranks.
   Args:
      x - a 1D numpy array
   Returns:
      array of midranks
   """
   J = np.argsort(x)
   Z = x[J]
   cumulative_weight = np.cumsum(sample_weight[J])
   N = len(x)
   T = np.zeros(N, dtype=np.float)
   i = 0
   while i < N:
       j = i
       while j < N and Z[j] == Z[i]:
           j += 1
       T[i:j] = cumulative_weight[i:j].mean()
       i = j
   T2 = np.empty(N, dtype=np.float)
   T2[J] = T
   return T2



def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight=None):

   if sample_weight is None:

       return fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)

   else:

       return fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight)





def fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight):

   """

   The fast version of DeLong's method for computing the covariance of

   unadjusted AUC.

   Args:

      predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]

         sorted such as the examples with label "1" are first

   Returns:

      (AUC value, DeLong covariance)

   Reference:

    @article{sun2014fast,

      title={Fast Implementation of DeLong's Algorithm for

             Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},

      author={Xu Sun and Weichao Xu},

      journal={IEEE Signal Processing Letters},

      volume={21},

      number={11},

      pages={1389--1393},

      year={2014},

      publisher={IEEE}

    }

   """

   # Short variables are named as they are in the paper

   m = label_1_count

   n = predictions_sorted_transposed.shape[1] - m

   positive_examples = predictions_sorted_transposed[:, :m]

   negative_examples = predictions_sorted_transposed[:, m:]

   k = predictions_sorted_transposed.shape[0]



   tx = np.empty([k, m], dtype=np.float)

   ty = np.empty([k, n], dtype=np.float)

   tz = np.empty([k, m + n], dtype=np.float)

   for r in range(k):

       tx[r, :] = compute_midrank_weight(positive_examples[r, :], sample_weight[:m])

       ty[r, :] = compute_midrank_weight(negative_examples[r, :], sample_weight[m:])

       tz[r, :] = compute_midrank_weight(predictions_sorted_transposed[r, :], sample_weight)

   total_positive_weights = sample_weight[:m].sum()

   total_negative_weights = sample_weight[m:].sum()

   pair_weights = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])

   total_pair_weights = pair_weights.sum()

   aucs = (sample_weight[:m]*(tz[:, :m] - tx)).sum(axis=1) / total_pair_weights

   v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights

   v10 = 1. - (tz[:, m:] - ty[:, :]) / total_positive_weights

   sx = np.cov(v01)

   sy = np.cov(v10)

   delongcov = sx / m + sy / n

   return aucs, delongcov





def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):

   """

   The fast version of DeLong's method for computing the covariance of

   unadjusted AUC.

   Args:

      predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]

         sorted such as the examples with label "1" are first

   Returns:

      (AUC value, DeLong covariance)

   Reference:

    @article{sun2014fast,

      title={Fast Implementation of DeLong's Algorithm for

             Comparing the Areas Under Correlated Receiver Oerating

             Characteristic Curves},

      author={Xu Sun and Weichao Xu},

      journal={IEEE Signal Processing Letters},

      volume={21},

      number={11},

      pages={1389--1393},

      year={2014},

      publisher={IEEE}

    }

   """

   # Short variables are named as they are in the paper

   m = label_1_count
   n = predictions_sorted_transposed.shape[1] - m
   positive_examples = predictions_sorted_transposed[:, :m]
   negative_examples = predictions_sorted_transposed[:, m:]
   k = predictions_sorted_transposed.shape[0]



   tx = np.empty([k, m], dtype=np.float)
   ty = np.empty([k, n], dtype=np.float)
   tz = np.empty([k, m + n], dtype=np.float)

   for r in range(k):

       tx[r, :] = compute_midrank(positive_examples[r, :])
       ty[r, :] = compute_midrank(negative_examples[r, :])
       tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

   aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
   v01 = (tz[:, :m] - tx[:, :]) / n
   v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m

   sx = np.cov(v01)
   sy = np.cov(v10)
   delongcov = sx / m + sy / n

   return aucs, delongcov


def calc_pvalue(aucs, sigma):
   """Computes log(10) of p-values.
   Args:
      aucs: 1D array of AUCs
      sigma: AUC DeLong covariances
   Returns:
      log10(pvalue)

   """

   l = np.array([[1, -1]])

   z = np.abs(np.diff(aucs)) / (np.sqrt(np.dot(np.dot(l, sigma), l.T)) + 1e-8)
   pvalue = 2 * (1 - st.norm.cdf(np.abs(z)))
   #  print(10**(np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)))
   return pvalue

def compute_ground_truth_statistics(ground_truth, sample_weight=None):
   assert np.array_equal(np.unique(ground_truth), [0, 1])
   order = (-ground_truth).argsort()
   label_1_count = int(ground_truth.sum())
   if sample_weight is None:
       ordered_sample_weight = None
   else:
       ordered_sample_weight = sample_weight[order]

   return order, label_1_count, ordered_sample_weight


def delong_roc_variance(ground_truth, predictions):
   """
   Computes ROC AUC variance for a single set of predictions
   Args:
      ground_truth: np.array of 0 and 1
      predictions: np.array of floats of the probability of being class 1
   """
   sample_weight = None
   order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
       ground_truth, sample_weight)
   predictions_sorted_transposed = predictions[np.newaxis, order]
   aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)

   assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
   return aucs[0], delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
   """
   Computes log(p-value) for hypothesis that two ROC AUCs are different
   Args:
      ground_truth: np.array of 0 and 1
      predictions_one: predictions of the first model,
         np.array of floats of the probability of being class 1
      predictions_two: predictions of the second model,
         np.array of floats of the probability of being class 1
   """
   sample_weight = None
   order, label_1_count,ordered_sample_weight = compute_ground_truth_statistics(ground_truth)
   predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
   aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count,sample_weight)

   return calc_pvalue(aucs, delongcov)

def delong_roc_ci(y_true,y_pred):
   aucs, auc_cov = delong_roc_variance(y_true, y_pred)
   auc_std = np.sqrt(auc_cov)
   lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
   ci = stats.norm.ppf(
       lower_upper_q,
       loc=aucs,
       scale=auc_std)
   ci[ci > 1] = 1
   return aucs,ci

#examples 具体用法
T1_DF = pd.read_csv(r'D:/data/AD/data/task12/T1/leaveoneout/MinMax/PCC/RFE_1/LR/cv_val_prediction.csv')
ASL_DF = pd.read_csv(r'D:/data/AD/data/task12/ASL/leaveoneout/Zscore/PCC/Relief_1/LR/cv_val_prediction.csv')
ASL_T1_DF = pd.read_csv(r'D:/data/AD/data/task12/ASL+T1/leaveoneout/Zscore/PCC/Relief_3/SVM/cv_val_prediction.csv')
Clinical_DF = pd.read_csv(r'D:/data/AD/data/task12/clinic/leaveoneout/MinMax/PCC/RFE_1/LR/cv_val_prediction.csv')
all_DF = pd.read_csv(r'D:/data/AD/data/task12/all/leaveoneout/Zscore/PCC/Relief_4/SVM/cv_val_prediction.csv')
preds_ASL = ASL_DF['Pred'].values
preds_T1 = T1_DF['Pred'].values
preds_ASLT1 = ASL_T1_DF['Pred'].values
preds_clinic = Clinical_DF['Pred'].values
preds_all = all_DF['Pred'].values
actual = ASL_DF['Label'].values
y_true= ASL_DF['Label'].values

alpha = .95

def delong_roc_ci(y_true,y_pred):
   aucs, auc_cov = delong_roc_variance(y_true, y_pred)
   auc_std = np.sqrt(auc_cov)
   lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
   ci = stats.norm.ppf(
       lower_upper_q,
       loc=aucs,
       scale=auc_std)
   ci[ci > 1] = 1
   return aucs,ci

#pvalue
pvalue = delong_roc_test(y_true,preds_ASL,preds_T1)
print('ASL VS T1:', pvalue)
pvalue = delong_roc_test(y_true,preds_ASLT1,preds_T1)
print('T1 VS T1+ASL:', pvalue)
pvalue = delong_roc_test(y_true,preds_clinic,preds_T1)
print('T1 VS Clinic:', pvalue)
pvalue = delong_roc_test(y_true,preds_all,preds_T1)
print('T1 VS ALL:', pvalue)
pvalue = delong_roc_test(y_true,preds_ASL,preds_ASLT1)
print('ASL VS T1+ASL:', pvalue)
pvalue = delong_roc_test(y_true,preds_ASL,preds_clinic)
print('ASL VS Clinic:', pvalue)
pvalue = delong_roc_test(y_true,preds_ASL,preds_all)
print('ASL VS all:', pvalue)
pvalue = delong_roc_test(y_true,preds_ASLT1,preds_clinic)
print('T1+ASL VS Clinic:', pvalue)
pvalue = delong_roc_test(y_true,preds_ASLT1,preds_all)
print('T1+ASL VS all:', pvalue)
pvalue = delong_roc_test(y_true,preds_clinic,preds_all)
print('Clinic VS all:', pvalue)
pvalue = delong_roc_test(y_true,preds_T1,preds_T1)
print(pvalue)
#  aucs, auc_cov = delong_roc_variance(y_true, y_pred)
# auc_1, auc_cov_1 = delong_roc_variance(y_true, y_pred_1)
# auc_2, auc_cov_2 = delong_roc_variance(y_true, y_pred_2)
#
# auc_std = np.sqrt(auc_cov_1)
# lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
# #
# ci = stats.norm.ppf(
#    lower_upper_q,
#    loc=auc_1,
#    scale=auc_std)
# ci[ci > 1] = 1
#
# print('95% AUC CI:', ci)
# print('AUC:', auc_1)




#DelongTest(preds_ASLT1, preds_ASL, actual)
