import pandas as pd
from pathlib import Path
import numpy as np
import re
import os

dirpath = r'\\219.228.149.7\syli\dataset\Primary and metastatic\zheci\Primary and metastatic OC\Primary and metastatic OC'
meta_list = [i for i in Path(dirpath).glob("*metastatic-seg")]
pri_list = [i for i in Path(dirpath).glob("*primary-seg")]
name_list, values_list = [], []
k = 1
for i in meta_list:
    for j in i.iterdir():
        name_list.append(j.name)
        values_list.append(k)
    k += 1

k = 1
for i in pri_list:
    for j in i.iterdir():
        name_list.append(j.name)
        values_list.append(k)
    k += 1

test_1_name = os.listdir(r'\\219.228.149.7\syli\dataset\Primary and metastatic\zheci\yfy Primary and metastatic-seg\4-yfy-metastatic-seg')
test_0_name = os.listdir(r'\\219.228.149.7\syli\dataset\Primary and metastatic\zheci\yfy Primary and metastatic-seg\4-yfy-Primary-seg')
test_name = test_1_name + test_0_name
test_batch = [4] * (len(test_1_name) + len(test_0_name))
name_list = name_list + test_name
values_list = values_list + test_batch
a = np.array([name_list, values_list]).T
df = pd.DataFrame(a, columns=['CaseName', 'batch'])
df.to_csv(r'C:\Users\Administrator\Desktop\ruanchaoliu\batch.csv', index=False, encoding="utf_8_sig")
