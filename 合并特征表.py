import os
import pandas as pd
number = 7
feature_path=['']*number
df_path=['']*number
select_df=['']*number
# feature_path[0]=r'D:/data/ovarioncus/0824/T1CE/log-sigma/firstorder/result\Zscore\PCC\RFE_1\RFE_train_feature.csv'
# df_path[0]=r'D:/data/ovarioncus/0824/T1CE/log-sigma/firstorder\T1CE_log-sigma_firstorder_train.csv'
# feature_path[1]=r'D:/data/ovarioncus/0824/T1CE/log-sigma/texture/result\MinMax/PCC\RFE_13\RFE_train_feature.csv'
# df_path[1]=r'D:/data/ovarioncus/0824/T1CE/log-sigma/texture/T1CE_log-sigma_Texture_test.csv'
# feature_path[2]=r'D:/data/ovarioncus/0824/T1CE/original/firstorder/result/Mean\PCC\Relief_5\Relief_train_feature.csv'
# df_path[2]=r'D:/data/ovarioncus/0824/T1CE/original/firstorder/T1CE_original_firstorder_test.csv'
feature_path[3]=r'D:/data/ovarioncus/0824/T1CE/original/shape/result/Zscore\PCC\RFE_13\RFE_train_feature.csv'
df_path[3]=r'D:/data/ovarioncus/0824/T1CE/original/shape/T1CE_original_shape_test.csv'
feature_path[4]=r'D:/data/ovarioncus/0824/T1CE/original/texture/result\Mean\PCC\Relief_2\Relief_train_feature.csv'
df_path[4]=r'D:/data/ovarioncus/0824/T1CE/original/texture/T1CE_original_Texture_test.csv'
feature_path[5]=r'D:/data/ovarioncus/0824/T1CE/wave/firstorder/result\Zscore\PCC\Relief_8\Relief_train_feature.csv'
df_path[5]=r'D:/data/ovarioncus/0824/T1CE/wave/firstorder/T1CE_wave_firstorder_test.csv'
# feature_path[6]=r'D:/data/ovarioncus/0824/T1CE/wave/texture/result\MinMax\PCC\Relief_10\Relief_train_feature.csv'
# df_path[6]=r'D:/data/ovarioncus/0824/T1CE/wave/texture/T1CE_wave_Texture_test.csv'

for i in [3,4,5]:
    print(i)
    feature_df=pd.read_csv(feature_path[i])
    feature=feature_df.columns[1:].values.tolist()   #索引行并转换为列表
    data_df=pd.read_csv(df_path[i])
    #data_df.rename(columns={'Unnamed: 0': 'CaseName'}, inplace=True)
    feature.insert(0, 'CaseName')
    #feature.insert(1, 'label')
    select_df[i]=data_df[feature]
    if i == 3:
        zong_df = select_df[i]
    else:
        zong_df = pd.merge(zong_df,select_df[i],on=('CaseName','label'))

if not os.path.exists(r'D:/data/ovarioncus/0824/T1CE/all'):
    os.makedirs(r'D:/data/ovarioncus/0824/T1CE/all/result')
zong_df.to_csv(r'D:/data/ovarioncus/0824/T1CE/all/test.csv', index=False)