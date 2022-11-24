import pandas as pd
excel_path = r'C:\Users\Administrator\Desktop\data.xlsx'
# df = pd.read_csv(excel_path)
T1_df = pd.read_excel(excel_path,sheet_name=0)
ASL_df = pd.read_excel(excel_path, sheet_name=1)
zong = pd.merge(T1_df, ASL_df, on=['subject', '分组','性别', '年龄', '受教育年限', '身高', '体重']) # , '分组','性别', '年龄', '受教育年限', '身高', '体重'
print(set(ASL_df['subject'].values.tolist()).difference(zong['subject'].values.tolist()))
new_df = pd.get_dummies(zong['分组'])
zong = pd.concat([new_df, zong], axis=1)
zong.to_csv(r'C:\Users\Administrator\Desktop\data_1.c',index=False,encoding="utf_8_sig")