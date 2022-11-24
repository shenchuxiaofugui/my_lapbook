from xpinyin import Pinyin
import pandas as pd
df = pd.read_excel(r'C:\Users\Administrator\Desktop\new_data\ASL.xlsx')
p = Pinyin()
pins = []
for i in list(df):
    pin = p.get_pinyin(i,'_') + '_ASL'
    pins.append(pin)
df.columns = pins
df.to_excel(r'C:\Users\Administrator\Desktop\new_data\ASL_pinyin.xlsx',index=False, encoding="utf_8_sig")
