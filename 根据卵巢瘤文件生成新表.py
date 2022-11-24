import pandas as pd
from pathlib import Path
import re
import os

def remove_chinese(word):
    new_word = re.sub('[\u4e00-\u9fa5]', '', word)
    new_word = re.sub('（）', '',new_word)
    new_word = re.sub(r"\(）", '', new_word)
    new_word = re.sub(r"（\)", '', new_word)
    return new_word

# data_df = pd.read_csv(r'\\219.228.149.7\syli\dataset\Primary and metastatic\zheci\yfy Primary and metastatic-seg\dataframe\dwi.csv')
# features = data_df['CaseName'].values.tolist()
# features = [remove_chinese(i) for i in features]
# data_df['CaseName'] = features
# data_df.to_csv(r'\\219.228.149.7\syli\dataset\Primary and metastatic\zheci\yfy Primary and metastatic-seg\dataframe\new\dwi.csv', index=False)
filepath = r'\\219.228.149.7\syli\dataset\Primary and metastatic\zheci\yfy Primary and metastatic-seg'
positive_file = [i for i in Path(filepath).glob('*metastatic-seg')]
negative_file = [i for i in Path(filepath).glob('*primary-seg')]
posi_case, nega_case = [], []
for i, j in zip(positive_file, negative_file):
    posi_case.extend(os.listdir(str(i)))
    nega_case.extend(os.listdir(str(j)))
posi_case = [remove_chinese(i) for i in posi_case]
nega_case = [remove_chinese(i) for i in nega_case]
posi_data = {"CaseName": posi_case, "label":[1] * len(posi_case)}
nega_data = {"CaseName": nega_case, "label":[0] * len(nega_case)}
posi_df = pd.DataFrame(posi_data)
nega_df = pd.DataFrame(nega_data)
df = pd.concat([posi_df, nega_df])
df.to_csv(r'\\219.228.149.7\syli\dataset\Primary and metastatic\zheci\yfy Primary and metastatic-seg\dataframe\all.csv', index=False)

