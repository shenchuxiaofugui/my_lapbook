from pathlib import  Path
import os
dirpath = '/homes/syli/dataset/zj_data/jly/gly1'
dirs = [i for i in Path(dirpath).iterdir()]
for i in dirs:
    os.rename(str(i), str(i).rstrip('0123456789'))