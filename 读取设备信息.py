import pandas as pd
from pathlib import Path

filepath = r'\\219.228.149.7\syli\dataset\zj_data\jly\data'
csv_header = ['magnetic']
info = pd.DataFrame(columns=csv_header)
for i in Path(filepath).iterdir():
    df = pd.read_csv(str(i) + '/dicom_magnetic_info.csv')
    station = df.iloc[0]
    station = station[1:].values.tolist()
    info.loc[i.name] = station
info.to_csv(r'C:\Users\Administrator\Desktop\ENT\eq2.csv')

# equment = pd.read_csv(r'C:\Users\Administrator\Desktop\ENT\equment.csv')
# df = pd.read_csv(r'C:\Users\Administrator\Desktop\ENT\temporary\data_1.csv')
# a = pd.merge(df, equment)
# a.to_csv(r'C:\Users\Administrator\Desktop\ENT\temporary\info.csv', index=False)