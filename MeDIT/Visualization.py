'''
MeDIT.Visualization
Functions for visualization.

author: Yang Song
All right reserved
'''

from __future__ import print_function
import numpy as np
from scipy.ndimage.morphology import binary_erosion
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.image import imread
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
import seaborn as sns


color_list = sns.color_palette(['#e50000', '#fffd01', '#87fd05', '#00ffff', '#152eff', '#ff08e8', '#ff5b00', '#9900fa']) \
             + sns.color_palette('deep')

from MeDIT.Normalize import Normalize01
from MeDIT.ArrayProcess import Index2XY

def FlattenImages(data_list, is_show=False, **kwargs):
    if len(data_list) == 1:
        return data_list[0]
    width = 1

    if data_list[0].ndim == 2:
        row, col = data_list[0].shape
        for one_data in data_list:
            temp_row, temp_col = one_data.shape
            assert(temp_row == row and temp_col == col)

        while True:
            if width * width >= len(data_list):
                break
            else:
                width += 1
        imshow_data = np.zeros((row * width, col * width))
        case_index = range(0, len(data_list))
        x, y = Index2XY(case_index, (width, width))

        for x_index, y_index, index in zip(x, y, case_index):
            imshow_data[x_index * row: (x_index + 1) * row, y_index * col: (y_index + 1) * col] = data_list[index]

        if is_show:
            plt.imshow(Normalize01(imshow_data), cmap='gray', **kwargs)
            plt.show()
        return imshow_data

    elif data_list[0].ndim == 3:
        row, col, slice = data_list[0].shape
        for one_data in data_list:
            temp_row, temp_col, temp_slice = one_data.shape
            assert (temp_row == row and temp_col == col and temp_slice == slice)

        while True:
            if width * width >= len(data_list):
                break
            else:
                width += 1
        imshow_data = np.zeros((row * width, col * width, slice))
        case_index = range(0, len(data_list))
        x, y = Index2XY(case_index, (width, width))

        for x_index, y_index, index in zip(x, y, case_index):
            imshow_data[x_index * row: (x_index + 1) * row, y_index * col: (y_index + 1) * col, :] = data_list[index]

        if is_show:
            Imshow3DArray(Normalize01(imshow_data))

        return imshow_data
    elif data_list[0].ndim == 4:
        slice, row, col, channel = data_list[0].shape
        for one_data in data_list:
            temp_slice, temp_row, temp_col, temp_channel = one_data.shape
            assert (temp_row == row and temp_col == col and temp_slice == slice and temp_channel == channel)

        while True:
            if width * width >= len(data_list):
                break
            else:
                width += 1
        imshow_data = np.zeros((slice, row * width, col * width, channel))
        case_index = range(0, len(data_list))
        x, y = Index2XY(case_index, (width, width))

        for x_index, y_index, index in zip(x, y, case_index):
            imshow_data[:, x_index * row: (x_index + 1) * row, y_index * col: (y_index + 1) * col, :] = \
                data_list[index]

        if is_show:
            Imshow3DArray(imshow_data)

        return imshow_data

def FlattenAllSlices(data, is_show=True):
    assert(data.ndim == 3)
    data_list = [data[..., index] for index in range(data.shape[-1] - 1)]
    return FlattenImages(data_list, is_show=is_show)


################################################################
# 该函数将每个2d图像进行变换。
def MergeImageWithRoi(data, roi, overlap=False, is_show=False):
    if data.max() > 1.0:
        print('Scale the data manually.')
        return data

    if data.ndim >= 3:
        print("Should input 2d image")
        return data

    if not isinstance(roi, list):
        roi = [roi]

    if len(roi) > len(color_list):
        print('There are too many ROIs')
        return data

    intensity = 255
    data = np.asarray(data * intensity, dtype=np.uint8)

    if overlap:
        merge_data = np.stack([data, data, data], axis=2)
        for one_roi, color in zip(roi, color_list[:len(roi)]):
            index_x, index_y = np.where(one_roi == 1)
            merge_data[index_x, index_y, :] = np.asarray(color) * intensity
    else:
        kernel = np.ones((3, 3))
        merge_data = np.stack([data, data, data], axis=2)
        for one_roi, color in zip(roi, color_list[:len(roi)]):
            boundary = one_roi - binary_erosion(input=one_roi, structure=kernel, iterations=1)
            index_x, index_y = np.where(boundary == 1)
            merge_data[index_x, index_y, :] = np.asarray(color) * intensity

    if is_show:
        plt.imshow(merge_data)
        plt.show()

    return merge_data

def  Merge3DImageWithRoi(data, roi, overlap=False):
    if not isinstance(roi, list):
        roi = [roi]

    merge_data = np.zeros((data.shape[2], data.shape[0], data.shape[1], 3))
    for slice_index in range(data.shape[2]):
        slice = data[..., slice_index]
        one_roi_list = []
        for one_roi in roi:
            one_roi_list.append(one_roi[..., slice_index])
        merge_data[slice_index, ...] = MergeImageWithRoi(slice, one_roi_list, overlap=overlap)

    return merge_data

def FusionImage(gray_array, fusion_array, color_map='jet', alpha=0.3, is_show=False):
    '''
    To Fusion two 2D images.
    :param gray_array: The background
    :param fusion_array: The fore-ground
    :param is_show: Boolen. If set to True, to show the result; else to return the fusion image. (RGB).
    :return:
    '''
    if gray_array.ndim >= 3:
        print("Should input 2d image")
        return gray_array

    dpi = 100
    x, y = gray_array.shape
    w = y / dpi
    h = x / dpi

    fig = plt.figure()
    fig.set_size_inches(w, h)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    plt.imshow(gray_array, cmap='gray')
    plt.imshow(fusion_array, cmap=color_map, alpha=alpha)

    fig.subplots_adjust(bottom=0)
    fig.subplots_adjust(top=1)
    fig.subplots_adjust(right=1)
    fig.subplots_adjust(left=0)

    plt.axis('off')
    plt.savefig('temp.jpg', format='jpeg', aspect='normal', bbox_inches='tight', pad_inches=0.0)
    merge_array = imread('temp.jpg')
    os.remove('temp.jpg')

    if is_show:
        plt.show()
    plt.close(fig)

    return merge_array

def ShowColorByRoi(background_array, fore_array, roi, color_map='OrRd', is_show=True):
    if background_array.shape != roi.shape:
        print('Array and ROI must have same shape')
        return

    background_array = Normalize01(background_array)
    fore_array = Normalize01(fore_array)
    cmap = plt.get_cmap(color_map)
    rgba_array = cmap(fore_array)
    rgb_array = np.delete(rgba_array, 3, 2)

    print(background_array.shape)
    print(rgb_array.shape)

    index_roi_x, index_roi_y = np.where(roi == 0)
    for index_x, index_y in zip(index_roi_x, index_roi_y):
        rgb_array[index_x, index_y, :] = background_array[index_x, index_y]

    plt.imshow(rgb_array)
    plt.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if is_show:
        plt.show()
    return rgb_array

def Imshow3DArray(data, roi=None, window_size=None, window_name='Imshow3D', overlap=False):
    '''
    Imshow 3D Array, the dimension is row x col x slice. If the ROI was combined in the data, the dimension is:
    slice x row x col x color
    :param data: 3D Array [row x col x slice] or 4D array [slice x row x col x RGB]
    '''
    if isinstance(roi, list) or isinstance(roi, type(data)):
        print(data.shape,roi.shape)
        data = Merge3DImageWithRoi(data, roi, overlap=overlap)

    if window_size is None:
        window_size = (800, 800)

    if np.ndim(data) == 3:
        data = np.swapaxes(data, 0, 1)
        data = np.transpose(data)

    pg.setConfigOptions(imageAxisOrder='row-major')
    app = QtGui.QApplication([])

    win = QtGui.QMainWindow()
    win.resize(window_size[0], window_size[1])
    imv = pg.ImageView()
    win.setCentralWidget(imv)
    win.show()
    win.setWindowTitle(window_name)

    imv.setImage(data)
    app.exec()

