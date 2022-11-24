'''
MeDIT.Statistics
Functions for process the numpy.array.

author: Yang Song, Ruiqi Yu, Yi-hong Zhang
All right reserved
'''

import os

import pandas as pd
import numpy as np
from scipy import stats
from sklearn import metrics
import SimpleITK as sitk
from sklearn.calibration import calibration_curve
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
import seaborn as sns

from MeDIT.ImageProcess import ReformatAxis
from MeDIT.Decorator import dict_decorator, figure_decorator

def MyWilcoxon(pred1, pred2):
    return stats.wilcoxon(pred1, pred2)[1]

def ComparePred(pred_list, name_list, func=MyWilcoxon):
    if len(pred_list) != len(name_list):
        raise Exception(r"pred_list length doesn't match the name_list length")

    case_num = len(pred_list)
    result_dict = {}
    for i in range(case_num):
        result_list = []
        for j in range(case_num):
            try:
                result = func(pred_list[i], pred_list[j])
            except ValueError:
                result = None
            result_list.append(result)
        result_dict[name_list[i]] = result_list

    result_df = pd.DataFrame(result_dict)
    result_df.index = name_list

    return result_df


def BoostSample(points, n_samples=1000):
    if isinstance(points, list):
        point_array = np.array(points)
    elif isinstance(points, pd.DataFrame):
        point_array = np.array(points)
    elif isinstance(points, pd.Series):
        point_array = points.values
    elif isinstance(points, np.ndarray):
        point_array = points
    else:
        print('The type of points is : ', type(points))

    samples = []
    for index in range(n_samples):
        one_sample = np.random.choice(point_array, size=point_array.size, replace=True)
        samples.append(one_sample.mean())

    return sorted(samples)

class BinaryClassification(object):
    def __init__(self, is_show=True, store_folder='', store_format='jpg'):
        self.color_list = sns.color_palette('deep')
        self._metric = {}
        self.UpdateShow(is_show)
        if os.path.isdir(store_folder):
            self.UpdateStorePath(store_folder, store_format)

    def UpdateShow(self, show):
        self._ConfusionMatrix.set_show(show)

        self._DrawRoc.set_show(show)
        self._DrawBox.set_show(show)
        self._DrawProbability.set_show(show)
        self._CalibrationCurve.set_show(show)

    def UpdateStorePath(self, store_folder, store_format='jpg'):
        self._ConfusionMatrix.set_store_path(os.path.join(store_folder, 'ConfusionMatrixInfo.csv'))

        self._DrawRoc.set_store_path(os.path.join(store_folder, 'ROC.{}'.format(store_format)))
        self._DrawBox.set_store_path(os.path.join(store_folder, 'Box.{}'.format(store_format)))
        self._DrawProbability.set_store_path(os.path.join(store_folder, 'Probability.{}'.format(store_format)))
        self._CalibrationCurve.set_store_path(os.path.join(store_folder, 'Calibration.{}'.format(store_format)))

    def __Auc(self, y_true, y_pred, ci_index=0.95):
        """
        This function can help calculate the AUC value and the confidence intervals. It is note the confidence interval is
        not calculated by the standard deviation. The auc is calculated by sklearn and the auc of the group are bootstraped
        1000 times. the confidence interval are extracted from the bootstrap result.

        Ref: https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291097-0258%2820000515%2919%3A9%3C1141%3A%3AAID-SIM479%3E3.0.CO%3B2-F
        :param y_true: The label, dim should be 1.
        :param y_pred: The prediction, dim should be 1
        :param CI_index: The range of confidence interval. Default is 95%
        """

        single_auc = metrics.roc_auc_score(y_true, y_pred)

        bootstrapped_scores = []

        np.random.seed(42)  # control reproducibility
        seed_index = np.random.randint(0, 65535, 1000)
        for seed in seed_index.tolist():
            np.random.seed(seed)
            pred_one_sample = np.random.choice(y_pred, size=y_pred.size, replace=True)
            np.random.seed(seed)
            label_one_sample = np.random.choice(y_true, size=y_pred.size, replace=True)

            if len(np.unique(label_one_sample)) < 2:
                continue

            score = metrics.roc_auc_score(label_one_sample, pred_one_sample)
            bootstrapped_scores.append(score)

        sorted_scores = np.array(bootstrapped_scores)
        std_auc, mean_auc = np.std(sorted_scores), np.mean(sorted_scores)

        ci = stats.norm.interval(ci_index, loc=mean_auc, scale=std_auc)
        return single_auc, mean_auc, std_auc, ci

    @dict_decorator()
    def _ConfusionMatrix(self, prediction, label, label_legend):
        prediction, label = np.array(prediction), np.array(label)
        self._metric['Sample Number'] = len(label)
        self._metric['Positive Number'] = np.sum(label)
        self._metric['Negative Number'] = len(label) - np.sum(label)

        fpr, tpr, threshold = metrics.roc_curve(label, prediction)
        index = np.argmax(1 - fpr + tpr)
        self._metric['Youden Index'] = threshold[index]

        pred = np.zeros_like(label)
        pred[prediction >= threshold[index]] = 1
        C = metrics.confusion_matrix(label, pred, labels=[1, 0])

        self._metric['accuracy'] = np.where(pred == label)[0].size / label.size
        if np.sum(C[0, :]) < 1e-6:
            self._metric['sensitivity'] = 0
        else:
            self._metric['sensitivity'] = C[0, 0] / np.sum(C[0, :])
        if np.sum(C[1, :]) < 1e-6:
            self._metric['specificity'] = 0
        else:
            self._metric['specificity'] = C[1, 1] / np.sum(C[1, :])
        if np.sum(C[:, 0]) < 1e-6:
            self._metric['PPV'] = 0
        else:
            self._metric['PPV'] = C[0, 0] / np.sum(C[:, 0])
        if np.sum(C[:, 1]) < 1e-6:
            self._metric['NPV'] = 0
        else:
            self._metric['NPV'] = C[1, 1] / np.sum(C[:, 1])

        single_auc, mean_auc, std, ci = self.__Auc(label, prediction, ci_index=0.95)
        self._metric['AUC'] = single_auc
        self._metric['95 CIs Lower'], self._metric['95 CIs Upper'] = ci[0], ci[1]
        self._metric['AUC std'] = std
        return self._metric

    @figure_decorator()
    def _DrawProbability(self, prediction, label, youden_index=0.5):
        df = pd.DataFrame({'prob': prediction, 'label': label})
        df = df.sort_values('prob')
        for prob_val, label_val in zip(df['prob'], df['label']):
            print(prob_val, label_val)
        bar_color = [self.color_list[x] for x in df['label'].values]
        print(len(bar_color))
        plt.figure(num=3, figsize=(8, 3))
        positive_label = []
        positive_val = []
        Negtive_label = []
        Negtive_val = []
        yindex = youden_index
        for index in range(len(prediction)):
            if df['label'].values[index] > 0:
                positive_label.append(index)
                positive_val.append(df['prob'].values[index] - yindex)
            else:
                Negtive_label.append(index)
                Negtive_val.append(df['prob'].values[index] - yindex)
        # plt.bar(range(len(prediction)), df['prob'].values - youden_index, color=bar_color)
        plt.bar(Negtive_label, Negtive_val, color=self.color_list[0], label="Normal")
        plt.bar(positive_label, positive_val, color=self.color_list[1], label="PAS disorders")
        print(youden_index)
        # plt.title("T2 validation nomogram")
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.ylabel("Prediction Probability")
        plt.xticks([])
        plt.yticks([0 - youden_index,
                    youden_index - youden_index,
                    1 - youden_index],
                   ['{:.2f}'.format(0),
                    '{:.3f}'.format(youden_index),
                    '{:.2f}'.format(1)
                    ])
        # plt.yticks([df['prob'].values.min() - youden_index,
        #             youden_index - youden_index,
        #             df['prob'].max() - youden_index],
        #            ['{:.2f}'.format(df['prob'].values.min()),
        #             '{:.2f}'.format(youden_index),
        #             '{:.2f}'.format(df['prob'].max())
        #             ])
        plt.legend()
        plt.savefig(r'C:\Users\HJ Wang\Desktop\waterfall.jpg', dpi=600)


    @figure_decorator()
    def _DrawBox(self, prediction, label, label_legend):
        prediction, label = np.array(prediction), np.array(label)
        positive = prediction[label == 1]
        negative = prediction[label == 0]

        sns.boxplot(data=[negative, positive])
        plt.xticks([0, 1], label_legend)

    @figure_decorator()
    def _DrawRoc(self, prediction, label):
        fpr, tpr, threshold = metrics.roc_curve(label, prediction)
        auc = metrics.roc_auc_score(label, prediction)
        name = 'AUC = {:.3f}'.format(auc)

        plt.plot(fpr, tpr, color=self.color_list[0], label=name, linewidth=3)

        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.05)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")

    @figure_decorator()
    def _CalibrationCurve(self, prediction, label):
        F, threshold = calibration_curve(label, prediction, n_bins=10)
        clf_score = metrics.brier_score_loss(label, prediction, pos_label=1)
        plt.plot(threshold, F, "s-", label='{:.3f}'.format(clf_score))
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.ylabel("Fraction of positives")
        plt.ylim([-0.05, 1.05])
        plt.legend(loc="lower right")

    def Run(self, pred, label, label_legend=('Negative', 'Positive'), store_folder=r''):
        assert(isinstance(pred, list))
        assert(isinstance(label, list))

        # self._ConfusionMatrix(pred, label, label_legend)
        # self._DrawRoc(pred, label)
        # self._DrawProbability(pred, label, youden_index=self._metric['Youden Index'])
        self._DrawProbability(pred, label, youden_index=0.516)

        # self._DrawBox(pred, label, label_legend)
        # self._CalibrationCurve(pred, label)


class BinarySegmentation(object):
    def __init__(self, store_folder=r'', is_show=True):
        self._metric = {}
        self.show = is_show
        self.store_path = os.path.join(store_folder, 'metric.csv')
        pass

    def _Dice(self, pred, label):
        smooth = 1
        intersection = (pred * label).sum()
        self._metric['Dice'] = (2 * intersection + smooth) / (pred.sum() + label.sum() + smooth)

    def _HausdorffDistanceImage(self, pred_image, label_image):
        hausdorff_computer = sitk.HausdorffDistanceImageFilter()
        hausdorff_computer.Execute(pred_image, label_image)
        self._metric['HD Image'] = hausdorff_computer.GetHausdorffDistance()

    def _HausdorffDistance(self, pred, label):
        pred_image = sitk.GetImageFromArray(pred)
        label_image = sitk.GetImageFromArray(label)

        hausdorff_computer = sitk.HausdorffDistanceImageFilter()
        hausdorff_computer.Execute(pred_image, label_image)
        self._metric['HD'] = hausdorff_computer.GetHausdorffDistance()

    def _ConfusionMatrix(self, pred, label):
        C = metrics.confusion_matrix(label.flatten(), pred.flatten(), labels=[1, 0])
        self._metric['accuracy'] = np.where(pred.flatten() == label.flatten())[0].size / label.flatten().size
        if np.sum(C[0, :]) < 1e-6:
            self._metric['sensitivity'] = 0
        else:
            self._metric['sensitivity'] = C[0, 0] / np.sum(C[0, :])
        if np.sum(C[1, :]) < 1e-6:
            self._metric['specificity'] = 0
        else:
            self._metric['specificity'] = C[1, 1] / np.sum(C[1, :])
        if np.sum(C[:, 0]) < 1e-6:
            self._metric['PPV'] = 0
        else:
            self._metric['PPV'] = C[0, 0] / np.sum(C[:, 0])
        if np.sum(C[:, 1]) < 1e-6:
            self._metric['NPV'] = 0
        else:
            self._metric['NPV'] = C[1, 1] / np.sum(C[:, 1])

    def ShowMetric(self):
        if self.show:
            print(self._metric)

    def SaveMetric(self):
        if self.store_path and self.store_path.endswith('csv'):
            df = pd.DataFrame(self._metric, index=[0])
            df.to_csv(self.store_path)

    def Run(self, pred, label):
        # Image类型的相关计算
        if isinstance(pred, sitk.Image) and isinstance(pred, sitk.Image):
            self._HausdorffDistanceImage(pred, label)
            ref = ReformatAxis()
            pred = ref.Run(pred)
            label = ref.Run(label)

        assert(isinstance(pred, np.ndarray) and isinstance(label, np.ndarray))
        assert(np.unique(pred).size == 2 and np.unique(label).size == 2)
        assert(pred.shape == label.shape)

        self._Dice(pred, label)
        self._HausdorffDistance(pred, label)

        self.ShowMetric()
        self.SaveMetric()


def DrawCalibrationCurve(csv):
    df = pd.read_csv(csv, index_col=0)
    y_pred = np.array(df["Pred"].values)
    y_label = np.array(df["Label"].values)
    bc = BinaryClassification()
    bc._CalibrationCurve(y_pred, y_label)


if __name__ == '__main__':
    DrawCalibrationCurve(r"D:/data/AD/data/task13/ASL+T1/leaveoneout/Zscore/PCC/RFE_4/SVM/cv_val_prediction.csv")
    DrawCalibrationCurve(r"D:/data/AD/data/task23/ASL+T1/leaveoneout/MinMax/PCC/RFE_4/LR/cv_val_prediction.csv")
    DrawCalibrationCurve(r"D:/data/AD/data/task12/ASL+T1/leaveoneout/Zscore/PCC/Relief_3/SVM/cv_val_prediction.csv")
    # pred1 = [0.346668, 0.503766, 0.485559, 0.49081, 0.564188, 0.542857, 0.447202, 0.386564, 0.528657, 0.471104]
    # pred2 = [0.149651, 0.476197, 0.413696, 0.493291, 0.947957, 0.802401, 0.46789, 0.044634, 0.725948, 0.361422]
    # pred3 = [0.202809, 0.608661, 0.540197, 0.256821, 0.676909, 0.709443, 0.376296, 0.040494, 0.562205, 0.391581]
    # pred4 = [0.112646, 0.5, 0.618102, 0.56328, 0.941308, 0.742813, 0.308826, 0.016663, 0.674971, 0.644663]
    # preds = [pred1, pred2, pred3, pred4]
    # names = ['model1', 'model2', 'model3', 'model4']

    # print(ComparePred(preds, names))
    # df = pd.read_csv(r'E:\research\gurouliu\bug_fix\imgscore_pselect_cli_reorder\MinMax\PCC\Relief_4\LR\test_prediction.csv')
    # df = pd.read_csv(r'C:\Users\HJ Wang\Desktop\ML\My_work\210322placenta\radiomics\20220104 DL feature\18 result\MinMax\PCC\ANOVA_5\LR\test_prediction.csv')
    # pred = df['Pred'].values.squeeze().tolist()
    # label = df['Label'].values.astype(int).squeeze().tolist()
    # #
    # metric = BinaryClassification(is_show=True)
    # metric.Run(pred, label)

  #  pred, label = np.zeros((200, 200)), np.zeros((200, 200))
  #  pred[100:150, 50:150] = 1
  #  label[100:150, 100:150] = 1

   # metric = BinarySegmentation(is_show=True)
   # metric.Run(pred, label)



