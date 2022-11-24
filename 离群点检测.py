import pandas as pd
df = pd.read_csv(r'C:\Users\Administrator\Desktop\ENT\CLAHE\ADC\new\train.csv')
geshu = [0] *200
for i in list(df)[4:]:
    mean_value = df[i].mean()
    std_values = df[i].std()
    liqun = [k for k in range(len(df[i])) if (df[i][k] < mean_value - std_values *3 or df[i][k]  >mean_value + std_values *3)]
    for j in liqun:
        geshu[j] += 1
a = [geshu.index(k) for k in geshu if k > 50]
for i in a:
    print(i, geshu[i])