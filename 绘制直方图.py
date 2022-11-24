import numpy as np  # numpy是Python中科学计算的核心库
import pandas as pd
import matplotlib.pyplot as plt  # matplotlib数据可视化神器


data_path = r'D:\data\ovarioncus\T1\result\Zscore\PCC\Relief_5\Relief_'
store_path = r'D:\data\ovarioncus\T1\FeatureViolin\new'

def normfun(x, mu, sigma):

    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

    return pdf

if __name__ == '__main__':

    train_data = pd.read_csv(data_path + 'train_feature.csv')  # 载入数据文件
    test_data = pd.read_csv(data_path + 'test_feature.csv')
    feature_list = list(train_data)
    feature_list = feature_list[2:]
    for i in feature_list:

        feature = train_data[i]  # 获得长度数据集
        test_feature = test_data[i]
        feature = pd.concat([feature, test_feature], axis=1)

        plt.hist(feature.T, bins=12, rwidth=0.9, density=True, label=['train', 'test'])

        plt.title(f'{i} distribution')

        plt.xlabel(i)

        plt.ylabel('Probability')
        plt.savefig(store_path + f'/{i}.jpg')
        plt.show()