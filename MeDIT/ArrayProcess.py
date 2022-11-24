'''
MeDIT.ArrayProcess
Functions for process the numpy.array.

author: Yang Song
All right reserved
'''

from copy import deepcopy
import numpy as np
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy import ndimage

### Roi Related ###################################################################
def SmoothRoi(roi, smooth_kernel=np.zeros((1, )), smooth_range=2):
    if smooth_kernel.sum() == 0:
        if len(np.shape(roi)) == 2:
            smooth_kernel = np.ones((3, 3))
        elif len(np.shape(roi)) == 3:
            smooth_kernel = np.ones((3, 3, 3))
        else:
            print('Only could process 2D or 3D data')
            return []

    inner_roi = deepcopy(roi)
    for index in range(smooth_range // 2):
        inner_roi = binary_erosion(input=inner_roi, structure=smooth_kernel, iterations=1)

    smooth_roi = np.zeros_like(roi)

    for index in range(smooth_range + 1):
        smooth_roi += 1. / (smooth_range + 1) * inner_roi
        inner_roi = binary_dilation(input = inner_roi, structure=smooth_kernel, iterations=1)

    return smooth_roi

def ExtractBoundaryOfRoi(roi):
    '''
    Find the Boundary of the binary mask. Which was used based on the dilation process.
    :param roi: the binary mask
    :return:
    '''
    kernel = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    boundary = binary_dilation(input=roi, structure=kernel, iterations=1) - roi
    return boundary

def RemoveSmallRegion(roi, size_threshold=50):
    # seperate each connected ROI
    label_im, nb_labels = ndimage.label(roi)

    new_roi = deepcopy(roi)
    # remove small ROI
    for i in range(1, nb_labels + 1):
        if (label_im == i).sum() < size_threshold:
            # remove the small ROI in mask
            new_roi[label_im == i] = 0
    return new_roi

def KeepLargestRoi(mask):
    if mask.max() == 0:
        return mask
    label_im, nb_labels = ndimage.label(mask)
    max_volume = [(label_im == index).sum() for index in range(1, nb_labels + 1)]
    index = np.argmax(max_volume)
    new_mask = np.zeros(mask.shape)
    new_mask[label_im == index + 1] = 1
    return new_mask

def GetRoiRange(roi, target_value=1):
    if not (roi.ndim == 2 or roi.ndim == 3):
        print('The dim of ROI should be 2 or 3')
        return None

    if np.ndim(roi) == 2:
        x, y = np.where(roi == target_value)
        x = np.unique(x)
        y = np.unique(y)
        return [x.tolist(), y.tolist()]
    else:
        x, y, z = np.where(roi == target_value)
        x = np.unique(x)
        y = np.unique(y)
        z = np.unique(z)
        return [x.tolist(), y.tolist(), z.tolist()]

### Transfer index to position #######################################################################################
def Index2XY(index, data_shape):
    '''
    Transfer the index to the x, y index based on the 2D image shape.
    :param index: The index list
    :param data_shape: The shape of the image.
    :return: The list of the x, y index.
    '''

    if np.max(index) >= data_shape[0] * data_shape[1]:
        print('The index is out of the range.')
        return []

    y = np.mod(index, data_shape[1])
    x = np.floor_divide(index, data_shape[1])
    return [x, y]

def XY2Index(position, data_shape):
    '''
    Transfer the x, y position to the index if flatten the 2D image.
    :param position: the point index with x and y
    :param data_shape: The shape of the image
    :return: the index of the flatted 1D vector.
    '''
    return position[0] * data_shape[1] + position[1]

def Index2XYZ(index, data_shape):
    '''
    Transfer the index to the x, y, z index based on the 3D image shape.
    :param index: The index index
    :param data_shape: The shape of the image.
    :return: The list of the x, y, z index.
    '''
    if np.max(index) >= data_shape[0] * data_shape[1] * data_shape[2]:
        print('The index is out of the range.')
        return []

    z = np.mod(index, data_shape[2])
    y = np.mod(np.floor_divide((index - z), data_shape[2]), data_shape[1])
    x = np.floor_divide(index, data_shape[2] * data_shape[1])
    return [x, y, z]

def XYZ2Index(position, data_shape):
    '''
    Transfer the x, y, z position to the index if flatten the 3D image.
    :param position: the point index with x and y
    :param data_shape: The shape of the image
    :return: the index of the flatted 1D vector.
    '''
    return position[0] * (data_shape[1] * data_shape[2]) + position[1] * data_shape[2] + position[2]

### Extract Patch from the image #######################################################################################
def ExtractPatch(array, patch_size, center_point=None, is_shift=True):
    '''
    Extract patch from a 2D image.
    :param array: the 2D numpy array
    :param patch_size: the size of the 2D patch
    :param center_point: the center position of the patch
    :param is_shift: If the patch is too close to the edge of the image, is it allowed to shift the patch in order to
    ensure that extracting the patch close to the edge. Default is True.
    :return: the extracted patch.
    '''

    if not center_point:
        center_point = [-1, -1]
    center_point = list(center_point)
    patch_size = np.asarray(patch_size)
    if patch_size.shape == () or patch_size.shape == (1,):
        patch_size = np.array([patch_size[0], patch_size[0]])

    image_row, image_col = np.shape(array)
    if patch_size[0] // 2 == image_row - (patch_size[0] // 2):
        catch_x_index = [patch_size[0] // 2]
    else:
        catch_x_index = np.arange(patch_size[0] // 2, image_row - (patch_size[0] // 2))
    if patch_size[1] // 2 == image_col - (patch_size[1] // 2):
        catch_y_index = [patch_size[1] // 2]
    else:
        catch_y_index = np.arange(patch_size[1] // 2, image_col - (patch_size[1] // 2))

    if center_point == [-1, -1]:
        center_point[0] = image_row // 2
        center_point[1] = image_col // 2

    if patch_size[0] > image_row or patch_size[1] > image_col:
        print('The patch_size is larger than image shape')
        return np.array([])

    if center_point[0] < catch_x_index[0]:
        if is_shift:
            center_point[0] = catch_x_index[0]
        else:
            print('The center point is too close to the negative x-axis')
            return []
    if center_point[0] > catch_x_index[-1]:
        if is_shift:
            center_point[0] = catch_x_index[-1]
        else:
            print('The center point is too close to the positive x-axis')
            return []
    if center_point[1] < catch_y_index[0]:
        if is_shift:
            center_point[1] = catch_y_index[0]
        else:
            print('The center point is too close to the negative y-axis')
            return []
    if center_point[1] > catch_y_index[-1]:
        if is_shift:
            center_point[1] = catch_y_index[-1]
        else:
            print('The center point is too close to the positive y-axis')
            return []

    x = [center_point[0] - patch_size[0] // 2, center_point[0] + patch_size[0] - patch_size[0] // 2]
    y = [center_point[1] - patch_size[1] // 2, center_point[1] + patch_size[1] - patch_size[1] // 2]

    patch = deepcopy(array[x[0]:x[1], y[0]:y[1]])
    return patch, [x, y]

def ExtractBlock(array, patch_size, center_point=None, is_shift=True):
    '''
    Extract patch from a 3D image.
    :param array: the 3D numpy array
    :param patch_size: the size of the 3D patch
    :param center_point: the center position of the patch
    :param is_shift: If the patch is too close to the edge of the image, is it allowed to shift the patch in order to
    ensure that extracting the patch close to the edge. Default is True.
    :return: the extracted patch.
    '''
    center_point = list(center_point)
    if not center_point:
        center_point = [-1, -1, -1]

    if not isinstance(center_point, list):
        center_point = list(center_point)
    patch_size = np.asarray(patch_size)
    if patch_size.shape == () or patch_size.shape == (1,):
        patch_size = np.array([patch_size[0], patch_size[0], patch_size[0]])

    image_row, image_col, image_slice = np.shape(array)
    if patch_size[0] // 2 == image_row - (patch_size[0] // 2):
        catch_x_index = [patch_size[0] // 2]
    else:
        catch_x_index = np.arange(patch_size[0] // 2, image_row - (patch_size[0] // 2))
    if patch_size[1] // 2 == image_col - (patch_size[1] // 2):
        catch_y_index = [patch_size[1] // 2]
    else:
        catch_y_index = np.arange(patch_size[1] // 2, image_col - (patch_size[1] // 2))
    if patch_size[2] == image_slice:
        catch_z_index = [patch_size[2] // 2]
    else:
        catch_z_index = np.arange(patch_size[2] // 2, image_slice - (patch_size[2] // 2))

    if center_point == [-1, -1, -1]:
        center_point[0] = image_row // 2
        center_point[1] = image_col // 2
        center_point[2] = image_slice // 2

    if patch_size[0] > image_row or patch_size[1] > image_col or patch_size[2] > image_slice:
        print('The patch_size is larger than image shape')
        return np.array()

    if center_point[0] < catch_x_index[0]:
        if is_shift:
            center_point[0] = catch_x_index[0]
        else:
            print('The center point is too close to the negative x-axis')
            return np.array()
    if center_point[0] > catch_x_index[-1]:
        if is_shift:
            center_point[0] = catch_x_index[-1]
        else:
            print('The center point is too close to the positive x-axis')
            return np.array()
    if center_point[1] < catch_y_index[0]:
        if is_shift:
            center_point[1] = catch_y_index[0]
        else:
            print('The center point is too close to the negative y-axis')
            return np.array()
    if center_point[1] > catch_y_index[-1]:
        if is_shift:
            center_point[1] = catch_y_index[-1]
        else:
            print('The center point is too close to the positive y-axis')
            return np.array()
    if center_point[2] < catch_z_index[0]:
        if is_shift:
            center_point[2] = catch_z_index[0]
        else:
            print('The center point is too close to the negative z-axis')
            return np.array()
    if center_point[2] > catch_z_index[-1]:
        if is_shift:
            center_point[2] = catch_z_index[-1]
        else:
            print('The center point is too close to the positive z-axis')
            return np.array()
    #
    # if np.shape(np.where(catch_x_index == center_point[0]))[1] == 0 or \
    #     np.shape(np.where(catch_y_index == center_point[1]))[1] == 0 or \
    #     np.shape(np.where(catch_z_index == center_point[2]))[1] == 0:
    #     print('The center point is too close to the edge of the image')
    #     return []

    x = [center_point[0] - patch_size[0] // 2, center_point[0] + patch_size[0] - patch_size[0] // 2]
    y = [center_point[1] - patch_size[1] // 2, center_point[1] + patch_size[1] - patch_size[1] // 2]
    z = [center_point[2] - patch_size[2] // 2, center_point[2] + patch_size[2] - patch_size[2] // 2]

    block = deepcopy(array[x[0]:x[1], y[0]:y[1], z[0]:z[1]])
    return block, [x, y, z]

def Crop2DArray(array, shape):
    '''
    Crop the size of the image. If the shape of the result is smaller than the image, the edges would be cut. If the size
    of the result is larger than the image, the edge would be filled in 0.
    :param array: the 2D numpy array
    :param shape: the list of the shape.
    :return: the cropped image.
    '''
    assert (array.ndim == len(shape))

    if array.shape[0] >= shape[0]:
        center = array.shape[0] // 2
        cropped_array = array[center - shape[0] // 2: center - shape[0] // 2 + shape[0], :]
    else:
        cropped_array = np.zeros((shape[0], array.shape[1]))
        center = shape[0] // 2
        cropped_array[center - array.shape[0] // 2: center - array.shape[0] // 2 + array.shape[0], :] = array
    array = cropped_array

    if array.shape[1] >= shape[1]:
        center = array.shape[1] // 2
        cropped_array = array[:, center - shape[1] // 2: center - shape[1] // 2 + shape[1]]
    else:
        cropped_array = np.zeros((array.shape[0], shape[1]))
        center = shape[1] // 2
        cropped_array[:, center - array.shape[1] // 2: center - array.shape[1] // 2 + array.shape[1]] = array


    return cropped_array

def Crop3DArray(array, shape):
    '''
    Crop the size of the image. If the shape of the result is smaller than the image, the edges would be cut. If the size
    of the result is larger than the image, the edge would be filled in 0.
    :param array: the 3D numpy array
    :param shape: the list of the shape.
    :return: the cropped image.
    '''
    if array.shape[0] >= shape[0]:
        center = array.shape[0] // 2
        new_image = array[center - shape[0] // 2: center - shape[0] // 2 + shape[0], :, :]
    else:
        new_image = np.zeros((shape[0], array.shape[1], array.shape[2]))
        center = shape[0] // 2
        new_image[center - array.shape[0] // 2: center - array.shape[0] // 2 + array.shape[0], :, :] = array
    array = new_image

    if array.shape[1] >= shape[1]:
        center = array.shape[1] // 2
        new_image = array[:, center - shape[1] // 2: center - shape[1] // 2 + shape[1], :]
    else:
        new_image = np.zeros((array.shape[0], shape[1], array.shape[2]))
        center = shape[1] // 2
        new_image[:, center - array.shape[1] // 2: center - array.shape[1] // 2 + array.shape[1], :] = array
    array = new_image

    if array.shape[2] >= shape[2]:
        center = array.shape[2] // 2
        new_image = array[:, :, center - shape[2] // 2: center - shape[2] // 2 + shape[2]]
    else:
        new_image = np.zeros((array.shape[0], array.shape[1], shape[2]))
        center = shape[2] // 2
        new_image[:, :, center - array.shape[2] // 2: center - array.shape[2] // 2 + array.shape[2]] = array
    array = new_image

    return new_image



