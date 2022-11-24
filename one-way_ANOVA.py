import os

#可视化包
import seaborn as sns
from IPython.display import Image
import matplotlib.pyplot as plt
sns.set()
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
plt.rcParams['font.sans-serif'] = ['simhei']
plt.rcParams['font.serif'] = ['simhei']
sns.set_style("darkgrid",{"font.sans-serif":['simhei','Droid Sans Fallback']})

#数学包
from scipy import stats
import pandas as pd
import numpy as np

#统计检验
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import MultiComparison


# 封装 双变量-单因素方差分析
def my_oneWayAnova(df, cata_name, num_name, alpha_anova=0.05, alpha_tukey=0.05):
    df[cata_name] = df[cata_name].astype('str')

    s1 = df[cata_name]
    s2 = df[num_name]

    fml = num_name + '~C(' + cata_name + ')'

    model = ols(fml, data=df).fit()
    anova_table_1 = anova_lm(model, typ=2).reset_index()
    p1 = anova_table_1.loc[0, 'PR(>F)']

    # 输出 ： 是否相等【不等式序列】
    if p1 > alpha_anova:
        print('组间【无】显著差异')
    else:
        print('组间【有】显著差异')
        # 输出不等式

    # 输出： 统计结果表（均值，分位数，差异组）
    df_p1 = df.groupby([cata_name])[num_name].describe()

    # 输出： Tudey 多重比较
    mc = MultiComparison(df[num_name], df[cata_name])
    df_smry = mc.tukeyhsd(alpha=alpha_tukey).summary()
    m = np.array(df_smry.data)
    df_p2 = pd.DataFrame(m[1:], columns=m[0])

    # 输出 ：分类直接的大小差异显著性
    df_p1_sub = df_p1[['mean']].copy()
    df_p1_sub.sort_values(by='mean', inplace=True)

    output_list = []

    for x in range(1, len(df_p1_sub.index)):
        if (df_p2.loc[((df_p2.group1 == df_p1_sub.index[x - 1]) & (df_p2.group2 == df_p1_sub.index[x])) |
                      ((df_p2.group1 == df_p1_sub.index[x]) & (df_p2.group2 == df_p1_sub.index[x - 1])),
                      'reject'].iloc[0]) == "True":
            smb = '<'
        else:
            smb = '<='
        if x == 1:
            output_list.append(df_p1_sub.index[x - 1])
            output_list.append(smb)
            output_list.append(df_p1_sub.index[x])
        else:
            output_list.append(smb)
            output_list.append(df_p1_sub.index[x])
    out_sentence = ' '.join(output_list)
    print(out_sentence)

    # 输出： 箱线图
    # 分布可视化boxplot
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=cata_name, y=num_name, data=df)  # ,order=df_p1_sub.index

    return df_p1, df_p2

def one_way(df, values):
    g = df['label'].unique()
    print(g)
    args = []
    for i in list(g):
        args.append(df[df['label'] == i][values])
    f, p = stats.f_oneway(*args)
    print(f, p)


feature = 'weight'
df_0 = pd.read_excel(r'D:\data\AD\new_data\linchuang.xlsx')
df1, df2 = my_oneWayAnova(df_0, cata_name='label', num_name=feature)
print(df1)
print(df2)
one_way(df_0, feature)