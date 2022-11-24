#!/D:/anaconda python
# -*- coding: utf-8 -*-
# @Project : MyScript
# @FileName: IntervalHist.py
# @IDE: PyCharm
# @Time  : 2020/11/24 17:38
# @Author : Jing.Z
# @Email : zhangjingmri@gmail.com
# @Desc : ==============================================
# Life is Short I Use Python!!!                      ===
# ======================================================

import os

import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np


class IntervalHist:
    def __init__(self):
        self.data_list = []
        self.data_name_list = []
        self.store_path = ''
        self.all_data = []
        self.array_num = ''

    def load_data(self, array_list, name_list, store_path):
        self.data_list = array_list
        self.data_name_list = name_list
        self.array_num = len(array_list)
        if store_path:
            self.store_path = store_path
        self.all_data = np.concatenate(array_list, axis=0)

    def _count(self, count_dict, array):
        # count sample num in a range of bins
        count_list = []
        bins_list = list(count_dict.keys())
        for i in bins_list:
            low = count_dict[i][0]
            up = count_dict[i][1]

            if i != bins_list[-1]:
                condition_low = np.extract(array >= low, array)
                target = np.extract(condition_low < up, condition_low)
            else:
                condition_low = np.extract(array >= low, array)
                target = np.extract(condition_low <= up, condition_low)
            target_num = target.size
            count_list.extend([target_num])

        # check num
        print('array num: {} \n sub count list {} \n sub count list sum: {}'.format(len(array),
                                                                                    count_list, sum(count_list)))
        return count_list

    # @pysnooper.snoop()
    def run(self, bins, title, is_show=True):
        color_list = ['cornflowerblue', 'tomato', 'gold', 'seagreen', 'hotpink']

        range_list = [min(self.all_data), max(self.all_data)]

        # U-test
        u_statistic, pVal = stats.ks_2samp(self.data_list[0], self.data_list[1])
        if pVal < 0.05:
            plt.title(title+'\n'+'p value < 0.05')
        else:
            plt.title(title + '\n' + 'p value'+ '%.3f'%pVal)

        for index in range(len(self.data_list)):
            sub_array = self.data_list[index]
            sub_name = self.data_name_list[index]

            width = (range_list[1] - range_list[0]) / bins

            # generate bins num and count
            count_dict = {}
            for i in range(int(bins)):
                count_dict[i] = [range_list[0] + i * width, range_list[0] + i * width + width]
            count = self._count(count_dict, sub_array)

            x = np.linspace(range_list[0], range_list[1], bins)
            plt.bar(x + index*width*(1/int(self.array_num)), count, align='center', width=width*(1/int(self.array_num)), edgecolor='gray',color=color_list[index], label=sub_name)
            # plt.show()
        plt.legend(loc='upper right')

        if self.store_path:
            if not os.path.exists(self.store_path):
                os.mkdir(self.store_path)
            plt.savefig(os.path.join(self.store_path, title + '.jpg'))
        if is_show:
            plt.show()
        plt.close()


def TestSame():
    array1 = np.random.normal(0, 1, 20)
    array2 = array1
    array3 = array1

    stagger_hist = IntervalHist()
    stagger_hist.load_data([array1, array2, array3], name_list=['1', '2', '3'], store_path='')
    stagger_hist.run(10, 'test')


def TestDifferent():
    array1 = np.random.normal(0, 1, 20)
    array2 = np.random.normal(0, 2, 20)
    array3 = np.random.normal(0, 3, 20)
    array4 = np.random.normal(0, 4, 20)

    stagger_hist = IntervalHist()
    stagger_hist.load_data([array1, array2, array3, array4], name_list=['1', '2', '3', '4'], store_path='')
    stagger_hist.run(10, 'test')


def main():
    TestDifferent()


if __name__ == '__main__':
    main()