'''
MeDIT.Normalize
Functions for normalizing the numpy.array.

author: Yang Song
All right reserved
'''

import numpy as np
from MeDIT.ArrayProcess import XY2Index, XYZ2Index

def NormalizeForTensorflow(data):
    data = np.asarray(data)
    if len(np.shape(data)) == 4:
        dim = 2
    if len(np.shape(data)) == 5:
        dim = 3
        
    means = list()
    stds = list()
    for index in range(np.shape(data)[0]):
        temp_data = data[index, ...]
        if dim == 2:
            means.append(np.mean(temp_data, axis=(0, 1)))
            stds.append(np.std(temp_data, axis=(0, 1)))
        if dim == 3:
            means.append(np.mean(temp_data, axis=(0, 1, 2)))
            stds.append(np.std(temp_data, axis=(0, 1, 2)))
            
    means = np.asarray(means)
    stds = np.asarray(stds)
    
    if dim == 2:
        means = means[..., np.newaxis, np.newaxis]
        means = np.transpose(means, (0, 2, 3, 1))
        stds = stds[..., np.newaxis, np.newaxis]
        stds = np.transpose(stds, (0, 2, 3, 1))
    if dim == 3:
        means = means[..., np.newaxis, np.newaxis, np.newaxis]
        means = np.transpose(means, (0, 2, 3, 4, 1))
        stds = stds[..., np.newaxis, np.newaxis, np.newaxis]
        stds = np.transpose(stds, (0, 2, 3, 4, 1))
        stds[stds == 0] = 1
                
    data -= means
    data /= stds
    
    return data


def NormalizeForTorch(data):
    data = np.asarray(data)
    if len(np.shape(data)) == 4:
        dim = 2
    elif len(np.shape(data)) == 5:
        dim = 3
    else:
        raise TypeError

    means = list()
    stds = list()
    for index in range(np.shape(data)[0]):
        temp_data = data[index, ...]
        if dim == 2:
            means.append(np.mean(temp_data, axis=(1, 2)))
            stds.append(np.std(temp_data, axis=(1, 2)))
        if dim == 3:
            means.append(np.mean(temp_data, axis=(1, 2, 3)))
            stds.append(np.std(temp_data, axis=(1, 2, 3)))

    means = np.asarray(means)
    stds = np.asarray(stds)

    if dim == 2:
        means = means[..., np.newaxis, np.newaxis]
        stds = stds[..., np.newaxis, np.newaxis]
    if dim == 3:
        means = means[..., np.newaxis, np.newaxis, np.newaxis]
        stds = stds[..., np.newaxis, np.newaxis, np.newaxis]
        stds[stds == 0] = 1

    data -= means
    data /= stds

    return data

'''This function to normalize the data for each sample and each modaity seperately'''
def NormalizeForModality(data):
    data = np.asarray(data)
    modalites = data.shape[-1]
    samples = data.shape[0]

    for sample in range(samples):
        for modality in range(modalites):
            data[sample, ..., modality] -= np.mean(data[sample, ..., modality])
            data[sample, ..., modality] = np.divide(data[sample, ..., modality], np.std(data[sample, ..., modality]))

    return data

def NormalizeEachSlice01(data):
    new_data = np.asarray(data, dtype=np.float32)
    for slice_index in range(np.shape(new_data)[2]):
        new_data[:, :, slice_index] = new_data[:, :, slice_index] - np.min(new_data[:, :, slice_index])
        if np.max(new_data[:, :, slice_index]) > 0.001:
            new_data[:, :, slice_index] = np.divide(new_data[:, :, slice_index], np.max(new_data[:, :, slice_index]))

    return new_data

def Normalize01(data, clip_ratio=0.0):
    normal_data = np.asarray(data, dtype=np.float32)
    if normal_data.max() - normal_data.min() < 1e-6:
        return np.zeros_like(normal_data)

    if clip_ratio > 1e-6:
        data_list = data.flatten().tolist()
        data_list.sort()
        normal_data.clip(data_list[int(clip_ratio / 2 * len(data_list))], data_list[int((1 - clip_ratio / 2) * len(data_list))])

    normal_data = normal_data - np.min(normal_data)
    normal_data = normal_data / np.max(normal_data)
    return normal_data

def NormalizeZ(data):
    normal_data = data - np.mean(data)
    if np.std(normal_data) < 1e-6:
        print('Check Normalization')
        return normal_data
    else:
        return normal_data / np.std(normal_data)

def IntensityTransfer(data, target_max, target_min, raw_max=None, raw_min=None):
    normal_data = np.float32(data)
    if raw_min is None:
        raw_min = np.min(normal_data)
    if raw_max is None:
        raw_max = np.max(normal_data)
    assert(target_max >= target_min)

    raw_intensity_range = raw_max - raw_min
    target_intensity_range = target_max - target_min
    normal_data = normal_data * target_intensity_range / raw_intensity_range
    normal_data = normal_data - np.min(normal_data) + target_min
    return normal_data

def NormalizeZByRoi(data, roi):
    if np.max(roi) > 0:
        mean_value = np.mean(data[roi == 1])
        std_value = np.std(data[roi == 1])

        normal_data = data - mean_value
        normal_data = normal_data / std_value

        # if len(np.where(roi == 1)) == 2:
        #     x, y = np.where(roi == 1)
        #     index = XY2Index([x, y], roi.shape)
        #
        #     vec = data.flatten()
        #     vec = vec[index]
        #
        #     mean_value = np.mean(vec)
        #     std_value = np.std(vec)
        #
        #     normal_data = data - mean_value
        #     normal_data = normal_data / std_value
        # elif len(np.where(roi == 1)) == 3:
        #     x, y, z = np.where(roi == 1)
        #     index = XYZ2Index([x, y, z], roi.shape)
        #
        #     vec = data.flatten()
        #     vec = vec[index]
        #
        #     mean_value = np.mean(vec)
        #     std_value = np.std(vec)
        #
        #     normal_data = data - mean_value
        #     normal_data = normal_data / std_value
        # else:
        #     print('Only support 2D and 3D data.')
        #     return None
    else:
        normal_data = data - np.mean(data)
        normal_data = normal_data / np.std(data)

    return normal_data
