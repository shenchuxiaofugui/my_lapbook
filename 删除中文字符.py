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
dirpath = r'\\219.228.149.7\syli\dataset\Primary and metastatic\zheci\Primary and metastatic OC\Primary and metastatic OC\all'
# meta_list = [i for i in Path(dirpath).glob("*metastatic-seg")]
# pri_list = [i for i in Path(dirpath).glob("*primary-seg")]
for oldname in Path(dirpath).iterdir():
    newname = remove_chinese(str(oldname))
    os.rename(str(oldname), newname)