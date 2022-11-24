import pandas as pd
from scipy import stats
# df = pd.read_csv(r'D:\data\ENT\resegment\T1+T2+T1CE+b1000\train_prediction1.csv')
# T1_data_arr = df['T1']
# other_data_arr = df['T1CE']
# _, p_value = stats.mannwhitneyu(T1_data_arr, other_data_arr)
# pcc = stats.pearsonr(T1_data_arr, other_data_arr)
# print(pcc)

feature_df = pd.read_csv(r'D:\data\ENT\resegment\T2+T1CE+T1\result\Mean\PCC\RFE_8\RFE_train_feature.csv')
features = list(feature_df)[1:]
features.insert(0, 'CaseName')
# feature.insert(1, 'label')
print(features)
for data in ['train', 'test', 'external_test']:
    df = pd.read_csv(f'D:\data\ENT/resegment\T2+T1CE+T1/{data}.csv')
    df = df[features]
    df.to_csv(f'D:\data\ENT/resegment\T2+T1CE+T1+b1000/{data}.csv', index=False)




