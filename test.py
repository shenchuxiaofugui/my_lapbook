import pandas as pd
import os
from pathlib import Path


features = ["CaseName", "Pred", "Label"]
for model in Path(r"C:\Users\13110\Desktop\EENT").iterdir():
    if os.path.isdir(str(model)):
        internal_df = pd.read_csv(str(model) + "/internal_test.csv")[features]
        external_df = pd.read_csv(str(model) + "/new_external_test.csv")[features]
        df = pd.concat([internal_df, external_df])
        df.to_csv(str(model) + "/test.csv", index=False)
    #     external_df = external_df[external_df["CaseName"] != "huangcaiping"]
    #     external_df = external_df[external_df["CaseName"] != "luolihua"]
    #     new_df = external_df[external_df["CaseName"] != "ruanyuerong"]
    #     assert len(new_df) == 29, "wrong"
    #     new_df.to_csv(str(model) + "/new_external_test.csv", index=False)