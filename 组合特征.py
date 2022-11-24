import os
import pandas as pd
number = 4
feature_path=['']*number
df_path=['']*number
select_df=['']*number
feature_path[0]=r'D:\data\ovarioncus\0824\T1\all\result\Mean\PCC\Relief_9\Relief_train_feature.csv'
df_path[0]=r'D:\data\ovarioncus\0824\T1\all\test.csv'
feature_path[1]=r'D:\data\ovarioncus\0824\T2\all\result\MinMax\PCC\Relief_1\Relief_train_feature.csv'
df_path[1]=r'D:\data\ovarioncus\0824\T2\all\test.csv'
feature_path[2]=r'D:\data\ENT\resegment\T1CE\all\result\Mean\PCC\RFE_9\RFE_train_feature.csv'
df_path[2]=r'D:\data\ENT\resegment\T1CE\all\train.csv'
feature_path[3]=r'D:\data\ENT\resegment\ADC\all\result\MinMax\PCC\RFE_6\RFE_train_feature.csv'
df_path[3]=r'D:\data\ENT\resegment\ADC\all\outside_test.csv'

for i in [0,1]:
    print(i)
    feature_df=pd.read_csv(feature_path[i])
    feature=feature_df.columns[1:].values.tolist()   #索引行并转换为列表
    data_df=pd.read_csv(df_path[i])
    #feature = [j.replace('t1.nii', 'T1.nii') for j in feature]
    #feature = [j.replace('t2.nii', 'T2.nii') for j in feature]
    #feature = [j.replace('t1_c.nii', 'T1CE.nii') for j in feature]
    feature.insert(0, 'CaseName')
    #feature.insert(1, 'label')
    select_df[i]=data_df[feature]
    if i == 0:
        zong_df = select_df[i]
    else:
        zong_df = pd.merge(zong_df,select_df[i],on=('CaseName','label'))
# zong_df=pd.merge(select_df[0],select_df[1],on=('CaseName','label'))  #on可以是两列
# zong_df=pd.merge(zong_df,select_df[2],on=('CaseName','label'))
zong_df.to_csv(r'D:\data\ovarioncus\0824\T1+T2\test.csv', index=False, encoding='utf_8_sig')