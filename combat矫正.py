import pandas as pd
import os
from combat import combat
import re
modals = ['dwi', 't1', 'T1CE', 't2']
dirpath = r'C:\Users\Administrator\Desktop\ruanchaoliu'
batch_df = pd.read_csv(os.path.join(dirpath, 'batch.csv'))
batch_df['CaseName'] = batch_df['CaseName'].astype(dtype='str')
for modal in modals:
    df_path = os.path.join(dirpath, modal)
    train_df = pd.read_csv(os.path.join(df_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(df_path, 'test.csv'))
    sum_df = pd.concat([train_df, test_df])
    sum_df['CaseName'] = sum_df['CaseName'].astype(dtype='str')
    zong_df = pd.merge(sum_df, batch_df)
    dat = zong_df.iloc[:, 2:-1].T
    ebat = combat(dat, zong_df['batch'], None)
    new_df = pd.merge(zong_df.iloc[:, :2],ebat.T, left_index=True,right_index=True)
    new_train_df = new_df.iloc[:-12, :]
    new_test_df = new_df.iloc[-12:, :]
    new_path = df_path + '/combat/result'
    os.makedirs(new_path)
    new_train_df.to_csv(df_path + '/combat/train.csv', index=False)
    new_test_df.to_csv(df_path + '/combat/test.csv', index=False)

