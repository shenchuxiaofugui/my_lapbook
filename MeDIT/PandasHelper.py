import pandas as pd
import numpy as np


def ReplaceSpaceWithNa(df):
    return df.replace(r'^\s*$', np.nan, regex=True)