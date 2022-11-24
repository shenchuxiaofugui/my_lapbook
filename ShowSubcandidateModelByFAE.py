#!/D:/anaconda python
# -*- coding: utf-8 -*-
# @Project : FAE tools
# @FileName: ShowSubcandidateModelByFAE.py
# @IDE: PyCharm
# @Time  : 2020/2/27 11:48
# @Author : Jing.Z
# @Email : zhangjingmri@gmail.com
# @Desc : ==============================================
# Life is Short I Use Python!!!                      ===
# ======================================================
import pysnooper
import csv
import os
import pandas as pd

from pathlib import Path
from BC.FeatureAnalysis.Classifier import *
from BC.FeatureAnalysis.FeaturePipeline import FeatureAnalysisPipelines
from BC.Visualization.PlotMetricVsFeatureNumber import DrawCurve


def smooth(original_array):
    temp_arr = []
    for i in range(len(original_array)):

        if i == 0 or i == len(original_array)-1:
            temp_arr.append(original_array[i])
        else:
            val = (original_array[i-1]+original_array[i]+original_array[i+1])/3
            temp_arr.append(val)

    return temp_arr

class FAEModels:
    """

            This class could compare different feature combinations from **result folder of FAE**, and return \
            comparision with candidate model selected in validation.

        Notes:

            * Candidate model could selected in validation by highest AUC, or standard error with *constraint*.
            1. If set *constraint* of 0, it means candidate model would be the model with highest AUC in validation.
            #. If set *constraint* of 1, it means candidate model would be the model with 1-SE in validation.

            * Intersection of hype-parameters of FAE result folder. For example, model A with 20 feature number, \
            while model B with 40 feature number, the final comparision would include only 20 feature number.

        Examples:

            ::

                model_path_dict = {'ADC:r'./example/ADC/result','DWI:./example/DWI/result'}``
                fae_models = FAEModels()``
                fae_models.load_model_info(model_path_dict)``
                fae_models.run(std_C=1, store_path=r'./example')``

    """
    def __init__(self):
        self.model_store_dict = {}
        self._fae = FeatureAnalysisPipelines()
        self.sheet_dict = dict()
        self.model_candidate_dict = {}
        self.model_info = []
        self.model_dict = {}
        self.model_result_dict = {}
        self.save_path = ''

    def load_model_info(self, fae_model_path_dict):
        """

            Intersection of hype-parameters would be processed.

        Parameters
        ----------
        fae_model_path_dict : {str : path}, model name and corresponding path include *_result.csv*.

        """
        self.model_dict = fae_model_path_dict
        model_key_name = list(fae_model_path_dict.keys())

        for sub_model in model_key_name:

            train_auc_pd = pd.read_csv(os.path.join(self.model_dict[sub_model], 'train_results.csv'), index_col=0)
            val_auc_pd = pd.read_csv(os.path.join(self.model_dict[sub_model], 'cv_val_results.csv'), index_col=0)
            test_auc_pd = pd.read_csv(os.path.join(self.model_dict[sub_model], 'test_results.csv'), index_col=0)
            self.model_result_dict[sub_model] = {'val': val_auc_pd, 'train': train_auc_pd, 'test': test_auc_pd}

            # take intersecion of FAE process hype-parameters

            if len(self.model_info) == 0:
                self.model_info = self.get_model_info(self.model_dict[sub_model])
            else:
                for index in range(len(self.model_info)):
                    self.model_info[index] = list(set(self.model_info[index]).intersection(
                        self.get_model_info(self.model_dict[sub_model])[index]))

    @staticmethod
    def get_model_info(model_store_path):
        """load information of FAE model, hype-parameters of FAE include Normalizer, Feature dimension reduction,
            Feature selection, Feature number and Classifiers"""

        fae = FeatureAnalysisPipelines()
        fae.LoadPipelineInfo(model_store_path)

        normalizer = fae.GetNormalizerList()
        normalizer_list = [i.GetName() for i in normalizer]
        dimension_reducer = fae.GetDimensionReductionList()
        dimension_reducer_list = [i.GetName() for i in dimension_reducer]
        feature_selector = fae.GetFeatureSelectorList()
        feature_selector_list = [i.GetName() for i in feature_selector]
        feature_number = fae.GetFeatureNumberList()
        classifier = fae.GetClassifierList()
        feature_classifier_list = [i.GetName() for i in classifier]

        return [normalizer_list, dimension_reducer_list, feature_selector_list, feature_number, feature_classifier_list]

    def candidate_info(self, normalizer, dimension_reducer, feature_selector, feature_number_list, classifier, std_C,
                       feature_max):
        """

            Choose a candidate model from validation set in a certain normalizer, dimension reducer, feature selector,
        feature number, classifier and std with constraint.


        """
        feature_number_list = [int(i) for i in feature_number_list if int(i) <= feature_max]
        feature_number_list.sort()

        compare_pd = []
        for model_key in list(self.model_result_dict.keys()):

            val_auc_pd = self.model_result_dict[model_key]['val']
            sub_model_name = normalizer + '_' + dimension_reducer + '_' + feature_selector + '_' + classifier
            sub_model_name_list = [normalizer + '_' + dimension_reducer + '_' + feature_selector + '_' +
                                   str(i) + '_' + classifier for i in feature_number_list]

            # test array
            test_array = self.model_result_dict[model_key]['test'].loc[sub_model_name_list, 'AUC'].to_numpy()

            sub_model_auc_pd = val_auc_pd.loc[sub_model_name_list]
            auc_array = np.asarray(sub_model_auc_pd['AUC'])

            # # smooth array
            # auc_array = smooth(auc_array)

            max_auc = max(auc_array)

            auc_std_array = np.asarray(sub_model_auc_pd['Std'])
            max_auc_std = auc_std_array[np.argmax(auc_array)]

            # default highest auc in validation
            candidate_feature_number = np.argmax(max_auc) + 1
            feature_number_list = [int(i) for i in feature_number_list]

            # choose candidate model
            for feature_index in feature_number_list:
                detect_auc = max_auc - std_C*max_auc_std
                sub_auc = auc_array[feature_index-1]
                if sub_auc >= detect_auc:
                    candidate_feature_number = feature_index
                    break

            candidate_model_name = normalizer + '_' + dimension_reducer + '_' + feature_selector + \
                '_' + str(candidate_feature_number) + '_' + classifier

            # store AUC vs feature number figure.

            DrawCurve([int(i) for i in feature_number_list], [list(auc_array), list(test_array)],
                      [list(std_C*auc_std_array)],
                      xlabel='feature number',
                      ylabel='auc', title=model_key+'_'+sub_model_name, name_list=['cv_val', 'test'], one_se=True,
                      store_path=os.path.join(self.save_path, model_key+'_'+sub_model_name+'.jpg'), is_show=False)

            result_pd = self.candidate_result_pd(model_key, candidate_model_name)

            if len(compare_pd) == 0:
                compare_pd = result_pd
            else:
                compare_pd = pd.concat([compare_pd, result_pd], axis=0, join='inner')
        # print(compare_pd)
        return compare_pd

    def candidate_result_pd(self, model_key, candidate_model_name, show_metric_list=None):
        """ show model comparision metric """

        if not show_metric_list:
            show_metric_list = ['train_AUC', 'train_95% CIs',  'train_Acc', 'train_Sen', 'train_Spe',
                                'val_AUC', 'val_95% CIs',  'val_Acc', 'val_Sen', 'val_Spe',
                                'test_AUC', 'test_95% CIs',  'test_Acc', 'test_Sen', 'test_Spe']

        # candidate_result_list = ['auc', 'auc 95% CIs', 'accuracy', 'sensitivity', 'specificity']
        merged_result_pd = []
        for cohort_index in ['train', 'val', 'test']:
            sub_result_pd = self.model_result_dict[model_key][cohort_index].loc[[candidate_model_name]]

            rename_columns = {}
            for columns in sub_result_pd.columns.tolist():
                rename_columns[columns] = cohort_index + '_' + columns
            sub_result_pd.rename(columns=rename_columns, inplace=True)
            # print(sub_result_pd)

            if len(merged_result_pd) == 0:
                merged_result_pd = sub_result_pd

            else:
                merged_result_pd = pd.concat([merged_result_pd, sub_result_pd], axis=1, join='inner')

        result_pd = merged_result_pd[show_metric_list]
        model_name = result_pd.index.tolist()[0]


        result_pd.rename(index={model_name: model_key + '_' + model_name}, inplace=True)

        # result_pd[result_pd.ix[0, 'val_auc'] > result_pd.ix[0, 'test_auc']] \
        #     = str(result_pd['val_auc'])

        return result_pd


    # @pysnooper.snoop()
    def add_features(self,  root_folder, saved_pd):
        # ADC_Zscore_PCC_RFE_5_SVM
        index_list = saved_pd.index.to_list()
        feature_dict = {}
        feature_pd = pd.DataFrame()
        for sub_index in index_list:
            model_parts = sub_index.split('_')
            if 'Zscore' in sub_index:
                model_name = sub_index.split('_Z')[0]

            if '_MinMax' in sub_index:
                model_name = sub_index.split('_M')[0]
            normalization, fr,fs, fn, cls = model_parts[-5], model_parts[-4], model_parts[-3], model_parts[-2], model_parts[-1]
            # D:\hospital\BreastCancer\radiomics_feature\all_cases\sub_model\DWI_b2000\result\Zscore\PCC\ANOVA_14
            result_key = Path(self.model_dict[model_name]).name
            feature_info_path = os.path.join(root_folder, model_name, result_key,
                                             normalization, fr, fs+'_'+fn, 'feature_select_info.csv')

            with open(feature_info_path, newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row[0] == 'selected_feature':
                        feature_list = row[1:]

            feature_dict[sub_index] = [feature_list]

        feature_pd = pd.DataFrame.from_dict(feature_dict, orient='index', columns=['selected_feature_list'])

        merged_pd = pd.concat([saved_pd, feature_pd], axis=1)

        return merged_pd



    def run(self, root_folder, std_C, feature_max, save_path=''):
        """ choose candidate model with iterating hype-parameters of FAE in std_C """
        saved_pd = []
        normalizer_list = self.model_info[0]
        dimension_reducer_list = self.model_info[1]
        feature_selector_list = self.model_info[2]
        feature_number_list = self.model_info[3]
        feature_classifier_list = self.model_info[4]

        if save_path:
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            self.save_path = os.path.join(save_path, 'compare'+'_std'+str(std_C)+'Fnum'+str(feature_max))
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)

        for normalizer_index, normalizer in enumerate(normalizer_list):
            for dimension_reducer_index, dimension_reducer in enumerate(dimension_reducer_list):
                for feature_selector_index, feature_selector in enumerate(feature_selector_list):
                    for classifier_index, classifier in enumerate(feature_classifier_list):
                        compared_pd = self.candidate_info(normalizer, dimension_reducer, feature_selector,
                                                          feature_number_list,  classifier, std_C, feature_max)
        # iterate model
        #                 blank_index = normalizer + '_' + dimension_reducer + '_' + feature_selector \
        #                     + '_' + classifier
                        show_list = compared_pd.columns.tolist()
                        compared_pd = compared_pd.round(3)
                        # blank_array = np.full([1, len(show_list)], np.nan)
                        # blank_row = pd.DataFrame(blank_array, index=['nan'], columns=show_list)

                        if len(saved_pd) == 0:
                            saved_pd = compared_pd

                        else:
                            saved_pd = pd.concat([saved_pd, compared_pd], axis=0, join='outer')
        saved_pd = self.add_features(root_folder, saved_pd)
        saved_pd.to_csv(os.path.join(self.save_path, 'compare.csv'))
        print(saved_pd)

        # sored_pd = saved_pd.sort_index(by=["val_auc"], inplace=True)
        # print(saved_pd)


def run(root_folder, result_key, store_path):
    model_path_dict = {# 'ADC': os.path.join(root_folder, 'ADC', result_key),
                       # 'DWI_b0': os.path.join(root_folder, 'DWI_b0', result_key),
                       # 'DWI_b50': os.path.join(root_folder, 'DWI_b50', result_key),
                       # 'DWI_b1000': os.path.join(root_folder, 'DWI_b1000', result_key),
                       # 'DWI_b2000': os.path.join(root_folder, 'DWI_b2000', result_key),
                       'T2W': os.path.join(root_folder, 'T2W', result_key),
                       'T1W_pre': os.path.join(root_folder, 'T1W_pre', result_key),
                       'T1W_post90s': os.path.join(root_folder, 'T1W_post90s', result_key),
                       'T1W_post5min': os.path.join(root_folder, 'T1W_post5min', result_key),
                       'SI_slope': os.path.join(root_folder, 'SI_slope', result_key),
                       'SEP': os.path.join(root_folder, 'SEP', result_key),
                       'MSI': os.path.join(root_folder, 'MSI', result_key),
                       'ESER': os.path.join(root_folder, 'ESER', result_key),
                       'E_peak': os.path.join(root_folder, 'E_peak', result_key),
                       'E_initial': os.path.join(root_folder, 'E_initial', result_key),}

    fae_models = FAEModels()

    fae_models.load_model_info(model_path_dict)
    fae_models.run(root_folder, std_C=1, feature_max=20,
                   save_path=store_path)


if __name__ == '__main__':
    run(r'D:\hospital\BreastCancer\radiomics_feature\all_cases\sub_model',
               'result', r'D:\hospital\BreastCancer\radiomics_feature\all_cases\sub_model\FAE_model_compare20201014\20_1_rsl\with_train_feature')