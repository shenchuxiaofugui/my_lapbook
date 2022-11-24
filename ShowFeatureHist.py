#!/D:/anaconda python
# -*- coding: utf-8 -*-
# @Project : MyScript
# @FileName: ShowFeatureHist.py
# @IDE: PyCharm
# @Time  : 2020/11/24 19:00
# @Author : Jing.Z
# @Email : zhangjingmri@gmail.com
# @Desc : ==============================================
# Life is Short I Use Python!!!                      ===
# ======================================================

import os
from pathlib import Path
from MeDIT.Statistics import BinaryClassification
import pysnooper
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import glob
import matplotlib.pyplot as plt
import seaborn as sn
from Utility.IntervalHist import IntervalHist
#from ZJ.FAE_tools.StatisticsFeature import FeatureStatistic


class FeatureHist:
    def __init__(self):
        self.feature_pd = pd.DataFrame()
        self.label_list = []
        self.selected_features = []
        self.store_path = ''

        self.label_0_pd = pd.DataFrame()
        self.label_1_pd = pd.DataFrame()

    def load(self, feature_path, label_list, fae_model_path):
        self.feature_pd = pd.read_csv(feature_path, index_col=0)
        self.label_list = label_list
        self.load_feature(fae_model_path)
        self.label_0_pd = self.feature_pd.loc[self.feature_pd['label'] == 0]
        self.label_1_pd = self.feature_pd.loc[self.feature_pd['label'] == 1]

    def load_feature(self, fae_model_path):
        # D:\hospital\BreastCancer\radiomics_feature\all_cases\sub_model\pre\result\Zscore\PCC\RFE_9\SVM
        feature_path = os.path.join(fae_model_path, 'feature_select_info.csv')

        feature_pd = pd.read_csv(feature_path, index_col=0, header=1)
        self.selected_features = list(feature_pd)

        return self.selected_features
        # self.selected_features = [i.replace('pre', 'post') for i in self.selected_features ]

    def show_feature_hist(self, store_path=''):
        for feature_index in self.selected_features:

            interval_hist = IntervalHist()
            interval_hist.load_data([self.label_0_pd[feature_index].to_numpy(),
                                     self.label_1_pd[feature_index].to_numpy()], self.label_list, store_path)
            if '_reset' in feature_index:
                feature_index = feature_index.replace('_reset', '')

            if '_original' in feature_index:
                feature_index = feature_index.replace('_original', '')

            interval_hist.run(bins=40, title=feature_index)


def main():

    feature_path = r'S:\jzhang\Tongji\BM\dataset_info0.7alltest\ESER_1\ESER.csv'
    model_path = r'U:\jzhang\Tongji\BM\dataset_info0.7alltest\ESER_1\result\Zscore\PCC\RFE_4'
    feature_hist = FeatureHist()
    feature_hist.load(feature_path, ['Benign', 'Malignant'], model_path)
    feature_hist.show_feature_hist(store_path=r'S:\jzhang\Tongji\BM\dataset_info0.7alltest\ESER_1\datset')
    # stat(feature_path, store_path=r'D:\hospital\BreastCancer\radiomics_feature\ImgNormalization\non_clip_without_norm_km_all_shape_more\merged_MinMax\dce_other_newDWI\dataset')


# def stat(feature_path, store_path):
#     fs = FeatureStatistic()
#     fs.load_feature_path(feature_path)
#
#     fs.difference_between_labels(store_path, 'features')


#D:\hospital\BreastCancer\radiomics_feature\all_cases\sub_model\post\dataset_time
def extract_pre_feature_on_post(pre_feature_model_path, post_feature_path, store_path):
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    feature_hist = FeatureHist()
    selected_features = feature_hist.load_feature(pre_feature_model_path)
    selected_features = [i.replace('pre', 'post') for i in selected_features ]
    selected_features.insert(0, 'label')
    post_train_path = os.path.join(post_feature_path, 'train_numeric_feature.csv')
    sub_train_pd = pd.read_csv(post_train_path, index_col=0, header=0)[selected_features]
    sub_train_pd.to_csv(os.path.join(store_path, 'train_numeric_feature.csv'))

    post_test_path = os.path.join(post_feature_path, 'test_numeric_feature.csv')
    sub_test_pd = pd.read_csv(post_test_path, index_col=0, header=0)[selected_features]
    sub_test_pd.to_csv(os.path.join(store_path, 'test_numeric_feature.csv'))



def show_violin(feature, label, title, store_path):

    if type(feature) is not np.ndarray:
        feature = np.array(feature)
        label = np.array(label)
    # x is a name ndarray can split train and test
    x = np.array(['' for _ in range(len(label))])

    fig = go.Figure()
    fig.add_trace(go.Violin(x=x[label == 0],
                            y=feature[label == 0],
                            legendgroup='Benign', scalegroup='Benign', name='Benign',
                            side='negative', line_color='blue', width=1))
    fig.add_trace(go.Violin(x=x[label == 1],
                            y=feature[label == 1],
                            legendgroup='Malignant', scalegroup='Malignant', name='Malignant',
                            side='positive', line_color='orange', width=1))
    # side='positive' in right, side='negative' in left
    fig.update_traces(meanline_visible=True)
    fig.update_layout(title_text=title, violingap=0, violinmode='overlay',
                      height=800, width=1500,  # figure size
                      font=dict(size=18),  # word size
                      legend={'font': dict(size=25),  # legend word size
                              'itemsizing': 'constant', },  # legend img size
                      hoverlabel=dict(font=dict(size=20)))


    fig.write_image(store_path)
    # fig.show()




def show_feature_violin():
    feature_path = r'D:\hospital\BreastCancer\radiomics_feature\ImgNormalization\non_clip_without_norm_km_all_shape_more\merged_MinMax\dce_other_newDWI\dataset\merged_original_train_feature.csv'
    store_path = r'D:\hospital\BreastCancer\radiomics_feature\ImgNormalization\non_clip_without_norm_km_all_shape_more\merged_MinMax\dce_other_newDWI\dataset\train'
    feature_pd = pd.read_csv(feature_path, index_col=0)

    feature_list = feature_pd.columns.tolist()[1:]

    for i in feature_list:
        feature_array = feature_pd[i].to_numpy()
        label_array = feature_pd['label'].to_numpy()
        sub_store_path = os.path.join(store_path, i+'_violin.png')
        show_violin(feature_array, label_array, i, sub_store_path)


def show_pred_violin(model_path, store_path):
    pred_pd = pd.read_csv(model_path+'\\test_prediction.csv')

    pred = pred_pd['Pred'].to_numpy()
    label = pred_pd['Label'].to_numpy()

    if type(pred) is not np.ndarray:
        pred = np.array(pred)
        label = np.array(label)
    # x is a name ndarray can split train and test
    x = np.array(['' for _ in range(len(label))])

    fig = go.Figure()
    fig.add_trace(go.Violin(x=x[label == 0],
                            y=pred[label == 0],
                            legendgroup='Benign', scalegroup='Benign', name='Benign',
                            side='negative', line_color='blue', width=1))
    fig.add_trace(go.Violin(x=x[label == 1],
                            y=pred[label == 1],
                            legendgroup='Malignant', scalegroup='Malignant', name='Malignant',
                            side='positive', line_color='orange', width=1))
    # side='positive' in right, side='negative' in left
    fig.update_traces(meanline_visible=True)
    fig.update_layout(title_text='Prediction', violingap=0, violinmode='overlay',
                      height=800, width=1500,  # figure size
                      font=dict(size=18),  # word size
                      legend={'font': dict(size=25),  # legend word size
                              'itemsizing': 'constant', },  # legend img size
                      hoverlabel=dict(font=dict(size=20)))

    sub_store_path = os.path.join(store_path, 'pred_violin.png')
    fig.write_image(sub_store_path)
    # metric.Run(pred, label)



def show_violin_train_test(data_set_folder, store_folder):

    pointpos_male = [-0.9, -1.1, -0.6, -0.3]
    pointpos_female = [0.45, 0.55, 1, 0.4]
    show_legend = [True, False, False, False]

    train_data_path = glob.glob(data_set_folder+'\\*train*.csv')
    train_pd = pd.read_csv(os.path.join(data_set_folder, train_data_path[0]), index_col=0)

    test_data_path = glob.glob(data_set_folder+'\\*test*.csv')
    test_pd = pd.read_csv(os.path.join(data_set_folder, test_data_path[0]), index_col=0)

    feature_list = train_pd.columns.tolist()[1:]

    for i in feature_list:
        print(i)
        sub_train_pd = train_pd[['label', i]]
        sub_train_label = sub_train_pd['label'].to_numpy()
        x = np.array(['Train' for _ in range(len(sub_train_label))])
        fig = go.Figure()
        a = x[sub_train_label == 0]
        # norm_feature_pd = (sub_train_pd[i]-sub_train_pd[i].mean())/sub_train_pd[i].std()
        norm_train_feature_pd = (sub_train_pd[i] - sub_train_pd[i].min()) / (sub_train_pd[i].max() - sub_train_pd[i].min())
        print(norm_train_feature_pd)
        fig.add_trace(go.Violin(x=x[sub_train_label == 0],
                                y=norm_train_feature_pd[sub_train_pd['label'] == 0].to_numpy(),
                                legendgroup='Benign', scalegroup='Benign', name='Train Benign',
                                side='negative', line_color='blue', width=1, points=False,
                                pointpos=-0.9,scalemode='count'))
        fig.add_trace(go.Violin(x=x[sub_train_label == 1],
                                y=norm_train_feature_pd[sub_train_pd['label'] == 1].to_numpy(),
                                legendgroup='Malignant', scalegroup='Malignant', name='Train Malignant',
                                side='positive', line_color='orange', width=1, points=False,
                                pointpos=0.45,scalemode='count'))


        sub_test_pd = test_pd[['label', i]]
        sub_test_label = sub_test_pd['label'].to_numpy()
        x = np.array(['Test' for _ in range(len(sub_test_label))])
        norm_test_feature_pd = (sub_test_pd[i] - sub_test_pd[i].min()) / (sub_test_pd[i].max() - sub_test_pd[i].min())
        fig.add_trace(go.Violin(x=x[sub_test_label == 0],
                                y=norm_test_feature_pd[sub_test_pd['label'] == 0].to_numpy(),
                                legendgroup='Test', scalegroup='Benign', name='Test Benign',
                                side='negative', line_color='lightseagreen', width=1, points=False,
                                pointpos= -0.6,scalemode='count'))
        fig.add_trace(go.Violin(x=x[sub_test_label == 1],
                                y=norm_test_feature_pd[sub_test_pd['label'] == 1].to_numpy(),
                                legendgroup='Test', scalegroup='Malignant', name='Test Malignant',
                                side='positive', line_color='mediumpurple', width=1, points=False,
                                pointpos=1,scalemode='count'))

        # side='positive' in right, side='negative' in left

        feature_class_list = ['glcm', 'gldm', 'glrlm', 'gldm', 'ngtdm', 'firstorder']
        title = i
        for j in feature_class_list:
            if j in i:
                sq = i.split(j)[0].replace('_', '').upper()
                feature_class = j.upper()
                feature_name = i.split('_')[-1]
                title = sq + ' ' + feature_class + ' ' + feature_name

        fig.update_traces(meanline_visible=True)
        fig.update_layout(title={'text': title,
                                    'y':0.95,
                                    'x':0.45,
                                    'xanchor': 'center',
                                    'yanchor': 'top'},
                          violingap=0, violinmode='overlay',
                          height=800, width=1500,  # figure size
                          font=dict(size=18),  # word size
                          legend={'font': dict(size=25),  # legend word size
                                  'itemsizing': 'constant', },  # legend img size
                          hoverlabel=dict(font=dict(size=20)))

        # fig.show()
        #
        # sub_store_path = os.path.join(store_folder, i+'_violin.png')
        # fig.write_image(sub_store_path)
        # feature_train_0 = norm_train_feature_pd[sub_train_pd['label'] == 0].to_numpy()
        # feature_train_1 = norm_train_feature_pd[sub_train_pd['label'] == 1].to_numpy()
        #
        # feature_test_0 = norm_test_feature_pd[sub_test_pd['label'] == 0].to_numpy()
        # feature_test_1 = norm_test_feature_pd[sub_test_pd['label'] == 1].to_numpy()
        #
        # plt.subplot(221)
        # plt.title('Train 0')
        # plt.hist(feature_train_0, edgecolor='black', bins=20)
        # plt.subplot(222)
        # plt.title('Train 1')
        # plt.hist(feature_train_1, edgecolor='black', bins=20)
        # plt.subplot(223)
        # plt.title('Test 0')
        # plt.hist(feature_test_0, edgecolor='black', bins=20)
        # plt.subplot(224)
        # plt.title('Test 1')
        # plt.hist(feature_test_1, edgecolor='black', bins=20)
        # sub_store_path = os.path.join(store_folder, i + '_hist.pdf')
        # plt.savefig(sub_store_path)
        # plt.close()
        sub_store_path = os.path.join(store_folder, i+'_violin.pdf')
        fig.write_image(sub_store_path)
        # metric.Run(pred, label)


def show_violin_train_test_box(data_set_folder, model_folder, store_folder):
    #feature_path = Path(model_folder).parent / 'feature_select_info.csv'

    #feature_pd = pd.read_csv(str(feature_path), index_col=0, header=1)
    #selected_features = list(feature_pd)

    pointpos_male = [-0.9, -1.1, -0.6, -0.3]
    pointpos_female = [0.45, 0.55, 1, 0.4]
    show_legend = [True, False, False, False]

    train_data_path = glob.glob(data_set_folder+'\\*train*.csv')
    train_pd = pd.read_csv(os.path.join(data_set_folder, train_data_path[0]), index_col=0)

    test_data_path = glob.glob(data_set_folder+'\\*test*.csv')
    test_pd = pd.read_csv(os.path.join(data_set_folder, test_data_path[0]), index_col=0)
    selected_features = list(train_pd)[2:]
    all_pd = pd.concat([train_pd, test_pd], axis=0)

    feature_list = train_pd.columns.tolist()[1:]
    # selected_features.insert(0, 'label')
    # all_features_pd = all_pd[selected_features]
    # all_features_pd['label'].loc[all_features_pd['label'] == 0] = 'Benign'
    # all_features_pd['label'].loc[all_features_pd['label'] == 1] = 'Malignant'

    concat_all_feature_pd = []

    for i in selected_features:

        print(i)

        sub_train_pd = train_pd[['label', i]]
        sub_test_pd = test_pd[['label', i]]
        sub_all_pd = all_pd[['label', i]]

        sub_train_label = sub_train_pd['label'].to_numpy()
        sub_test_label = sub_test_pd['label'].to_numpy()
        sub_all_label = sub_all_pd['label'].to_numpy()

        x = np.array(['Train' for _ in range(len(sub_train_label))])

        # norm_train_feature_pd = (sub_train_pd[i] - sub_train_pd[i].min()) / (sub_train_pd[i].max() - sub_train_pd[i].min())
        norm_train_feature_pd = sub_train_pd[i]
        train_data = norm_train_feature_pd.to_numpy()
        merge_all_pd = pd.DataFrame()
        merge_all_pd['label'] = sub_all_label
        # merge_all_pd['feature value'] = all_pd[i].to_numpy()

        # norm_all_pd = (all_pd[i] - all_pd[i].min()) / (
        #             all_pd[i].max() - all_pd[i].min())
        norm_all_pd = all_pd[i]
        merge_all_pd['feature value'] = norm_all_pd.to_numpy()
        merge_all_pd['feature name'] = [i] * len(sub_all_label)

        if len(concat_all_feature_pd) == 0:
            concat_all_feature_pd = merge_all_pd
        else:
            concat_all_feature_pd = pd.concat([concat_all_feature_pd, merge_all_pd], axis=0)

        norm_train_pd = pd.DataFrame()
        norm_train_pd['label'] = sub_train_label
        norm_train_pd['Dataset'] = ['Training'] * len(sub_train_label)
        norm_train_pd[i] = train_data

        #
        # norm_test_feature_pd = (sub_test_pd[i] - sub_test_pd[i].min()) / (
        #             sub_test_pd[i].max() - sub_test_pd[i].min())
        norm_test_feature_pd = sub_test_pd[i]
        test_data = norm_test_feature_pd.to_numpy()
        norm_test_pd = pd.DataFrame()
        norm_test_pd['label'] = sub_test_label
        norm_test_pd['Dataset'] = ['Test'] * len(sub_test_label)
        norm_test_pd[i] = test_data
        labels = 'Train benign', 'Train malignant', 'Test benign', 'Test malignant'

        merged_pd = pd.concat([norm_train_pd, norm_test_pd],axis=0)
        plt.figure(dpi=600)
        merged_pd['label'] .loc[merged_pd['label'] == 0] = 'Benign'
        merged_pd['label'] .loc[merged_pd['label'] == 1] = 'Malignant'
        # print(merged_pd)

        my_pal = {"Benign": "lightskyblue", "Malignant": "dodgerblue"}

        sn.violinplot(data=merged_pd, x='Dataset', scale='area', y=i, palette='OrRd_r',
                      hue='label', split=True, inner="quartile", saturation=1)

        sub_store_path = os.path.join(store_folder, i + '_violin.pdf')
        plt.savefig(sub_store_path)
        sub_store_path = os.path.join(store_folder, i + '_violin.png')
        plt.savefig(sub_store_path)

        plt.close()

        merge_all_pd['label'] .loc[merge_all_pd['label'] == 0] = 'Benign'
        merge_all_pd['label'] .loc[merge_all_pd['label'] == 1] = 'Malignant'

        sn.violinplot(data=merge_all_pd,  x='feature name', y='feature value', scale='area', palette='OrRd_r',
                  hue='label', split=True, inner="quartile", saturation=1)
        # plt.show()
        sub_store_path = os.path.join(store_folder, i + '_violin_singlefeature_all.pdf')
        plt.savefig(sub_store_path)
        sub_store_path = os.path.join(store_folder, i + '_violin_singlefeature_all.png')
        plt.savefig(sub_store_path)
        plt.close()


    concat_all_feature_pd['label'] .loc[concat_all_feature_pd['label'] == 0] = 'Benign'
    concat_all_feature_pd['label'] .loc[concat_all_feature_pd['label'] == 1] = 'Malignant'
    sn.violinplot(data=concat_all_feature_pd, x='feature name', y='feature value', scale='area', palette='OrRd_r',
                  hue='label', split=True, inner="quartile", saturation=1)
    plt.close()





if __name__ == '__main__':
    # main()
    model_path = r'C:\Users\Administrator\Desktop\ENT\ADC\all\result\Zscore\PCC\RFE_4\LR'
    store_path = r'C:\Users\Administrator\Desktop\ENT\ADC\all'
    feature_path = r'S:\jzhang\Tongji\BM\dataset_random_again\all'
    # extract_pre_feature_on_post(model_path, feature_path, store_path)
    # show_feature_violin()

    data_set_folder = r'D:\data\ENT\resegment\b1000'
    model_flder = r'D:\data\ovarioncus\0824\DWI\all\result\MinMax\PCC\Relief_7\LR'
    store_folder = r'D:\data\ENT\resegment\b1000\feature'
    if not Path(store_folder).exists():
        Path(store_folder).mkdir()
    show_violin_train_test_box(data_set_folder, model_flder, store_folder)

    # model_path = r'S:\jzhang\Tongji\BM\dataset_random_again\all\result\Zscore\PCC\RFE_4\LR'
    # show_violin_train_test_box(r'S:\jzhang\Tongji\BM\dataset_random_again\all\result\Zscore\PCC\RFE_4',
    #                        r'S:\jzhang\Tongji\BM\dataset_random_again\all\result\Zscore\PCC\RFE_4')