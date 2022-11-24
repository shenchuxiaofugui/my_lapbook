import scipy.stats
import pandas as pd

path = r'D:\data\ovarioncus\T1\result\Zscore\PCC\Relief_5\Relief_'
train_data = pd.read_csv(path + 'train_feature.csv')  # 载入数据文件
test_data = pd.read_csv(path + 'test_feature.csv')
feature_list = list(train_data)
feature_list = feature_list[2:]
for i in feature_list:

    train_feature = train_data[i].values  # 获得长度数据集
    test_feature = test_data[i].values
    KL = scipy.stats.entropy(train_feature, test_feature)
    print(i, KL)