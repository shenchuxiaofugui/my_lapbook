import pandas as pd
excel_path = r'D:\data\AD\data\task13'
feature_df = pd.read_csv(excel_path + '/p_test_split_statistics.csv')
feature_df = feature_df[feature_df['p-value'] <= 0.05/3]
feature_list = feature_df['feature'].values.tolist()
#feature_list = [i for i in feature_list if '_ASL' not in i]
feature_list.insert(0, 'label')
feature_list.insert(0, 'CaseName')
print(len(feature_list))
df = pd.read_csv(excel_path + '/data.csv')
df = df[feature_list]
df.to_csv(excel_path + '\\ASLT1\\p_data.csv',index=False,encoding="utf_8_sig")