# -*- coding: utf-8 -*-
import numpy as np
import random
import os
import pandas as pd
from scipy import stats


def data_separate_by_label(total_data):
    if total_data.columns.isin(['label', 'Label']).any():
        label_column_name = 'label' if total_data.columns.isin(['label']).any() else 'Label'
    else:
        label_column_name = total_data.columns.to_list()[1]
    label_list = np.array(total_data[label_column_name].tolist())

    index0 = np.where(label_list == 0)[0]
    index1 = np.where(label_list == 1)[0]

    group0_dataframe = set_new_dataframe(total_data, index0)
    group1_dataframe = set_new_dataframe(total_data, index1)

    return group0_dataframe, group1_dataframe


# 给总数据df和拆分的索引列表，把新的df提取出来
def set_new_dataframe(dataframe, case_index):
    col_name = dataframe.columns.tolist()
    new_dataframe = pd.DataFrame()
    new_dataframe = new_dataframe.append(dataframe.loc[case_index, :], ignore_index=True)
    new_dataframe = new_dataframe[col_name]  # 不知道为什么遇到过顺序变乱的情况，这样重新调整顺序
    return new_dataframe


# 输入两组数据df，计算所有P-value
def p_test(train_df, test_df, alpha=1e-3):
    assert train_df.columns.tolist() == test_df.columns.tolist(), 'train and test feature mismatch'
    features = train_df.columns.tolist()[2:]
    p_list = []
    distribute = []
    for feature in features:
        train_data_arr = train_df[feature].values
        test_data_arr = test_df[feature].values
        print(feature)
        print('train mean/std', train_data_arr.mean(), train_data_arr.std())
        print('test mean/std', test_data_arr.mean(), test_data_arr.std())
        _, normal_p = stats.normaltest(np.concatenate((train_data_arr, test_data_arr), axis=0))
        if len(set(train_data_arr)) < 10 and feature != 'chanci':  # 少于5个数认为是离散值，用卡方检验
            p_value = p_test_categories(train_data_arr, test_data_arr)
            p_list.append(float('%.5f' % p_value))
            distribute.append('categories')
        elif normal_p > alpha:  # 正态分布用T检验
            _, p_value = stats.ttest_ind(train_data_arr, test_data_arr)
            p_list.append(float('%.5f' % p_value))
            distribute.append('normal')
        else:  # P很小，拒绝假设，假设是来自正态分布，非正态分布用u检验
            _, p_value = stats.mannwhitneyu(train_data_arr, test_data_arr)
            p_list.append(float('%.5f' % p_value))
            distribute.append('non-normal')
    return features, p_list, distribute


def p_test_categories(train_data_arr, test_data_arr):  # 用于对非连续值的卡方检验
    count1 = count_list(train_data_arr)
    count2 = count_list(test_data_arr)  # dict, 每个类别为key，统计了每个类别的次数
    categories = set(list(count1.keys()) + list(count2.keys()))
    contingency_dict = {}
    for category in categories:
        contingency_dict[category] = [count1[category] if category in count1.keys() else 0,
                                      count2[category] if category in count2.keys() else 0]

    contingency_pd = pd.DataFrame(contingency_dict)
    contingency_array = np.array(contingency_pd)
    _, p_value, _, _ = stats.chi2_contingency(contingency_array)
    return p_value


def count_list(input):
    if not isinstance(input, list):
        input = list(input)
    dict = {}
    for i in set(input):
        dict[i] = input.count(i)
    return dict


# excel_path = r'C:\Users\HJ Wang\Desktop\ML\My_work\210322placenta\临床.csv'
excel_path = r'D:\data\AD\data\clinic.csv'
df = pd.read_csv(excel_path)
# df = pd.read_excel(excel_path)
output_path = r'D:\data\AD\data'

# 先默认二分类
label_0, label_1 = data_separate_by_label(df)
feature_list, p_value_list, distribution = p_test(label_0, label_1)
for i in range(len(feature_list)):
    print('%s p-value is %.5f' % (feature_list[i], p_value_list[i]))
p_value_arr = np.array(p_value_list)
mean_p_value = p_value_arr.mean()
print('p-value average is', mean_p_value)
output_p_test_dict = {'feature': feature_list,
                      'p-value': p_value_list,
                      'distribution': distribution}
output_p_test_df = pd.DataFrame(output_p_test_dict)
output_p_test_df.to_csv(os.path.join(output_path, 'p_test_split_statistics.csv'), index=False, encoding="utf_8_sig")