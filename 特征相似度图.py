import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# df = pd.read_csv(r'C:\Users\HJ Wang\Desktop\model result T2+clinic\T2+C z-score feature\relief3 0.818AUC'
#                  r'\Relief_test_feature.csv')
# df = pd.read_csv(r'C:\Users\HJ Wang\Desktop\model result T2+clinic\T2+C z-score feature\relief3 0.818AUC\\'
#                  r'test_prediction.csv')
df = pd.read_csv(r'D:\data\AD\data\task13\ASL+T1\4feature.csv')
# # df = pd.read_csv(r'C:\Users\HJ Wang\Desktop\model result T2+clinic\T2+C z-score feature\两种子模型特征结合 0.839AUC\\'
#                  r'test_prediction.csv')
# df.sort_values(by='label', inplace=True, ascending=True)
# feature_list = ['rad-score', 'AGE', 'FIGO']
# feature_list = df.columns.tolist()[2:]
#feature_list = ['DWI.nii_original_glcm_Idmn','DWI.nii_original_glrlm_LowGrayLevelRunEmphasis','DWI.nii_original_glrlm_RunLengthNonUniformityNormalized','DWI.nii_wavelet-LHH_glcm_InverseVariance']
#data_array = df[feature_list].values
df1=df.iloc[:,2:]
#df1=df.iloc[:,2:]
df1=(df1-df1.mean())/df1.std()
data_array=df1.values

# w_df = pd.read_csv(r'C:\Users\HJ Wang\Desktop\model result T2+clinic\T2+C z-score feature\relief3 0.818AUC\LR_coef.csv')
# w = w_df['Coef'].values
# data_array = data_array * w
# data_array = np.sum(data_array, axis=1)
#print(data_array)

case_num = len(df['label'])
case_list = ['' for _ in range(case_num)]
for i in range(case_num):
    if i % 8 == 0:
        case_list[i] = i+1
case_list[-1] = case_num
target_array = np.zeros((case_num, case_num))
case_list2 = [case_list[-i] for i in range(1, case_num+1)]


def euclidean_distance(array1, array2):
    return np.sqrt(np.sum((array1 - array2)**2))


for i in range(case_num):
    for j in range(case_num):
        data1 = data_array[i]
        data2 = data_array[j]
        target_array[i][j] = euclidean_distance(data1, data2)   #计算欧式距离
        # 所有case之间的特征相关性算完了，然后画图


print(target_array)
f, ax = plt.subplots(figsize=(30, 30))
ax.set_title('Comprehensive  Nomogram Feature Similarities', fontsize=60)
h = sns.heatmap(np.flip(target_array, axis=0), cmap='YlOrRd_r', linewidths=0, vmin=0, cbar=False)

cb = h.figure.colorbar(h.collections[0])  # 显示color bar
cb.ax.tick_params(labelsize=50)

ax.set_xticklabels(case_list, fontsize=50)
ax.set_yticklabels(case_list2, fontsize=50)
ax.set_xlabel('Patients', fontsize=50)
ax.set_ylabel('Patients', fontsize=50)

plt.savefig(r'D:\data\AD\data\task13\ASL+T1\sns_heatmap_center1.jpg', dpi=300)
#plt.show()