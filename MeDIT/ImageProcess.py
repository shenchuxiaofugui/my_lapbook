'''
MeDIT.ImageProcess
Functions for process the SimpleITK Image.

author: Yang Song, Jing Zhang
All right reserved
'''

# import dicom2nifti
import pydicom
import os
import shutil
import SimpleITK as sitk
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.ndimage.morphology import binary_dilation, binary_erosion

def ProcessROIImage(roi_image, process, store_path='', is_2d=True):
    # Dilate or erode the roi image.
    # If the type of process is int, it denotes the voxel unit.
    # If the type of process is float, it denotes the percentage unit.

    _, roi = GetDataFromSimpleITK(roi_image, dtype=np.uint8)
    if roi.ndim != 3:
        print('Only process on 3D data.')
        return
    if np.max(roi) == 0:
        print('Not valid ROI!')
        return

    if isinstance(process, int):
        if is_2d:
            kernel = np.ones((3, 3))
            processed_roi = np.zeros_like(roi)
            for slice_index in range(roi.shape[2]):
                slice = roi[..., slice_index]
                if np.max(slice) == 0:
                    continue
                if process > 0:
                    processed_roi[..., slice_index] = binary_dilation(slice, kernel, iterations=np.abs(process)).astype(roi.dtype)
                else:
                    processed_roi[..., slice_index] = binary_erosion(slice, kernel, iterations=np.abs(process)).astype(roi.dtype)
        else:
            kernel = np.ones((3, 3, 3))
            if process > 0:
                processed_roi = binary_dilation(roi, kernel, iterations=np.abs(process)).astype(roi.dtype)
            else:
                processed_roi = binary_erosion(roi, kernel, iterations=np.abs(process)).astype(roi.dtype)
    elif isinstance(process, float):
        if is_2d:
            kernel = np.ones((3, 3))
            processed_roi = deepcopy(roi)
            for slice_index in range(roi.shape[2]):
                slice = deepcopy(roi[..., slice_index])
                if np.max(slice) == 0:
                    continue

                if np.abs(process) < 1e-6:
                    processed_roi[..., slice_index] = deepcopy(roi[..., slice_index])
                elif process > 1e-6:
                    while np.sum(processed_roi[..., slice_index]) / np.sum(slice) < 1 + process:
                        processed_roi[..., slice_index] = binary_dilation(slice, kernel, iterations=1).astype(roi.dtype)
                else:
                    while np.sum(processed_roi[..., slice_index]) / np.sum(slice) > 1 + process:
                        processed_roi[..., slice_index] = binary_erosion(processed_roi[..., slice_index], kernel, iterations=1).astype(roi.dtype)
        else:
            kernel = np.ones((3, 3, 3))
            processed_roi = deepcopy(roi)
            if np.abs(process) < 1e-6:
                processed_roi = deepcopy(roi)
            elif process > 1e-6:
                while np.sum(processed_roi) / np.sum(roi) < 1 + process:
                    processed_roi = binary_dilation(roi, kernel, iterations=1).astype(roi.dtype)
            else:
                while np.sum(processed_roi) / np.sum(roi) > 1 + process:
                    processed_roi = binary_erosion(processed_roi, kernel, iterations=1).astype(roi.dtype)
    else:
        processed_roi = roi
        print('The type of the process is not in-valid.')
        return sitk.Image()


    processed_roi_image = GetImageFromArrayByImage(processed_roi, roi_image)

    if store_path:
        sitk.WriteImage(processed_roi_image, store_path)
    return processed_roi_image

def GetImageFromArrayByImage(data, refer_image, is_transfer_axis=True, flip_log=[0, 0, 0]):
    if is_transfer_axis:
        data = np.swapaxes(data, 0, 1)
        for index, is_flip in enumerate(flip_log):
            if is_flip:
                data = np.flip(data, axis=index)
        data = np.transpose(data)
    

    new_image = sitk.GetImageFromArray(data)
    new_image.CopyInformation(refer_image)
    return new_image

def GetDataFromSimpleITK(image, dtype=np.float32):
    ref = ReformatAxis()
    data = ref.Run(image)
    return data.astype(dtype), ref

def _GenerateFileName(file_path, name):
    store_path = ''
    if os.path.splitext(file_path)[1] == '.nii':
        store_path = file_path[:-4] + '_' + name + '.nii'
    elif os.path.splitext(file_path)[1] == '.gz':
        store_path = file_path[:-7] + '_' + name + '.nii.gz'
    else:
        print('the input file should be suffix .nii or .nii.gz')

    return store_path

################################################################################
def ResizeSipmleITKImage(image, expected_resolution=None, expected_shape=None, method=sitk.sitkBSpline, dtype=sitk.sitkFloat32):
    '''
    Resize the SimpleITK image. One of the expected resolution/spacing and final shape should be given.

    :param image: The SimpleITK image.
    :param expected_resolution: The expected resolution.
    :param excepted_shape: The expected final shape.
    :return: The resized image.

    Apr-27-2018, Yang SONG [yang.song.91@foxmail.com]
    '''


    if (expected_resolution is None) and (expected_shape is None):
        print('Give at least one parameters. ')
        return image

    shape = image.GetSize()
    resolution = image.GetSpacing()

    if expected_resolution is None:
        dim_0, dim_1, dim_2 = False, False, False
        if expected_shape[0] < 1e-6:
            expected_shape[0] = shape[0]
            dim_0 = True
        if expected_shape[1] < 1e-6:
            expected_shape[1] = shape[1]
            dim_1 = True
        if expected_shape[2] < 1e-6:
            expected_shape[2] = shape[2]
            dim_2 = True
        expected_resolution = [raw_resolution * raw_size / dest_size for dest_size, raw_size, raw_resolution in
                               zip(expected_shape, shape, resolution)]
        if dim_0: expected_resolution[0] = resolution[0]
        if dim_1: expected_resolution[1] = resolution[1]
        if dim_2: expected_resolution[2] = resolution[2]
        
    elif expected_shape is None:
        dim_0, dim_1, dim_2 = False, False, False
        if expected_resolution[0] < 1e-6: 
            expected_resolution[0] = resolution[0]
            dim_0 = True
        if expected_resolution[1] < 1e-6: 
            expected_resolution[1] = resolution[1]
            dim_1 = True
        if expected_resolution[2] < 1e-6: 
            expected_resolution[2] = resolution[2]
            dim_2 = True
        expected_shape = [int(raw_resolution * raw_size / dest_resolution) for
                       dest_resolution, raw_size, raw_resolution in zip(expected_resolution, shape, resolution)]
        if dim_0: expected_shape[0] = shape[0]
        if dim_1: expected_shape[1] = shape[1]
        if dim_2: expected_shape[2] = shape[2]

    # output = sitk.Resample(image, expected_shape, sitk.AffineTransform(len(shape)), method, image.GetOrigin(),
    #                        expected_resolution, image.GetDirection(), dtype)
    resample_filter = sitk.ResampleImageFilter()
    output = resample_filter.Execute(image, expected_shape, sitk.AffineTransform(len(shape)), method, image.GetOrigin(),
                           expected_resolution, image.GetDirection(), 0.0, dtype)
    return output

def ResizeNiiFile(file_path, store_path='', expected_resolution=None, expected_shape=None, method=sitk.sitkBSpline, dtype=sitk.sitkFloat32):
    expected_resolution = deepcopy(expected_resolution)
    expected_shape = deepcopy(expected_shape)
    if not store_path:
        store_path = _GenerateFileName(file_path, 'Resize')

    image = sitk.ReadImage(file_path)
    resized_image = ResizeSipmleITKImage(image, expected_resolution, expected_shape, method=method, dtype=dtype)
    sitk.WriteImage(resized_image, store_path)

def ResizeRoiNiiFileByRef(file_path, ref_image, store_path=''):
    if isinstance(ref_image, str):
        ref_image = sitk.ReadImage(ref_image)
    expected_shape = ref_image.GetSize()
    if not store_path:
        store_path = _GenerateFileName(file_path, 'Resize')
    image = sitk.ReadImage(file_path)
    resized_image = ResizeSipmleITKImage(image, expected_shape=expected_shape, method=sitk.sitkLinear, dtype=sitk.sitkFloat32)
    data = sitk.GetArrayFromImage(resized_image)

    new_data = np.zeros(data.shape, dtype=np.uint8)
    new_data[data > 0.5] = 1
    new_image = sitk.GetImageFromArray(new_data)
    new_image.CopyInformation(resized_image)
    sitk.WriteImage(new_image, store_path)

################################################################################
def RegistrateImage(fixed_image, moving_image, interpolation_method=sitk.sitkBSpline):
    '''
    Registrate SimpleITK Imageby default parametes.

    :param fixed_image: The reference
    :param moving_image: The moving image.
    :param interpolation_method: The method for interpolation. default is sitkBSpline
    :return: The output image

    Apr-27-2018, Jing ZHANG [798582238@qq.com],
                 Yang SONG [yang.song.91@foxmail.com]
    '''
    if isinstance(fixed_image, str):
        fixed_image = sitk.ReadImage(fixed_image)
    if isinstance(moving_image, str):
        moving_image = sitk.ReadImage(moving_image)

    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                          moving_image,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                  sitk.Cast(moving_image, sitk.sitkFloat32))
    output_image = sitk.Resample(moving_image, fixed_image, final_transform, interpolation_method, 0.0,
                                     moving_image.GetPixelID())
    return output_image

def RegistrateNiiFile(fixed_image_path, moving_image_path, interpolation_method=sitk.sitkBSpline):
    output_image = RegistrateImage(fixed_image_path, moving_image_path, interpolation_method)
    store_path = _GenerateFileName(moving_image_path, 'Reg')
    sitk.WriteImage(output_image, store_path)

def GetTransformByElastix(fix_image_path, moving_image_path, output_folder,
                          elastix_folder=r'D:\MyCode\Lib\Elastix',
                          parameter_folder=r'D:\MyCode\Lib\Elastix\RegParam\3ProstateBspline16'):
    '''
    Get registed transform by Elastix. This is depended on the Elastix.

    :param elastix_folder: The folder path of the built elastix.
    :param fix_image_path: The path of the fixed image.
    :param moving_image_path: The path of the moving image.
    :param output_folder: The folder of the output
    :param parameter_folder: The folder path that store the parameter files.
    :return:
    '''
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    cmd = os.path.join(elastix_folder, 'elastix') + r' -f "' + fix_image_path + r'" -m "' + moving_image_path + r'" -out "' + output_folder + '"'
    file_path_list = os.listdir(parameter_folder)
    file_path_list.sort()
    for file_path in file_path_list:
        abs_file_path = os.path.join(parameter_folder, file_path)
        cmd += r' -p "' + abs_file_path + '"'
    os.system(cmd)

def RegisteByElastix(moving_image_path, transform_folder, elastix_folder=r'D:\MyCode\Lib\Elastix'):
    '''
    Registed Image by Elastix. This is depended on the Elastix.

    :param elastix_folder: The folder path of the built Elastix
    :param moving_image_path: The path of the moving image
    :param transform_folder: The folder path of the generated by the elastix fucntion.
    :return:
    '''
    file_name, suffex = os.path.splitext(moving_image_path)

    temp_folder = os.path.join(transform_folder, 'temp')
    try:
        os.mkdir(temp_folder)
    except:
        pass
    try:
        cmd = os.path.join(elastix_folder, 'transformix') + r' -in "' + moving_image_path + r'" -out "' + temp_folder + '"'
        candidate_transform_file_list = os.listdir(transform_folder)
        candidate_transform_file_list.sort()
        for file_path in candidate_transform_file_list:
            if len(file_path) > len('Transform'):
                if 'Transform' in file_path:
                    abs_transform_path = os.path.join(transform_folder, file_path)
                    cmd += r' -tp "' + abs_transform_path + '"'

        os.system(cmd)
    except:
        shutil.rmtree(temp_folder)

    try:
        shutil.move(os.path.join(temp_folder, 'result.nii'), file_name + '_Reg' + suffex)
        shutil.rmtree(temp_folder)
    except:
        pass

    try:
        shutil.move(os.path.join(temp_folder, 'result.hdr'), file_name + '_Reg' + '.hdr')
        shutil.move(os.path.join(temp_folder, 'result.img'), file_name + '_Reg' + '.img')
        shutil.rmtree(temp_folder)
    except:
        pass

    try:
        shutil.move(os.path.join(temp_folder, 'result.mhd'), file_name + '_Reg' + '.mhd')
        shutil.move(os.path.join(temp_folder, 'result.raw'), file_name + '_Reg' + '.raw')
        shutil.rmtree(temp_folder)
    except:
        pass

    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)

#########################################################################

def FindNfitiDWIConfigFile(file_path, is_allow_vec_missing=True):
    if file_path.endswith('.nii.gz'):
        file_name = file_path[:-7]
        dwi_file = file_name + '.nii.gz'
    else:
        file_name = os.path.splitext(file_path)[0]
        dwi_file = file_name + '.nii'

    dwi_bval_file = file_name + '.bval'
    dwi_vec_file = file_name + '.bvec'

    if os.path.exists(dwi_file) and os.path.exists(dwi_bval_file):
        if os.path.exists(dwi_vec_file):
            return dwi_file, dwi_bval_file, dwi_vec_file
        else:
            if is_allow_vec_missing:
                return dwi_file, dwi_bval_file, ''
            else:
                print('Check these files')
                return '', '', ''
    else:
        print('Check these files')
        return '', '', ''

def SeparateNfitiDWIFile(dwi_file_path):
    if dwi_file_path.endswith('.nii.gz'):
        suffex = '.nii.gz'
    elif dwi_file_path.endswith('.nii'):
        suffex = '.nii'

    dwi_file, dwi_bval_file, _ = FindNfitiDWIConfigFile(dwi_file_path)
    if dwi_file and dwi_bval_file:
        dwi_4d = nb.load(dwi_file)

        with open(dwi_bval_file, 'r') as b_file:
            bvalue_str = b_file.read()[:-1]
        if ' ' in bvalue_str:
            bvalue_list = bvalue_str.split(' ')
        elif '\t' in bvalue_str:
            bvalue_list = bvalue_str.split('\t')


        dwi_list = nb.funcs.four_to_three(dwi_4d)
        if len(dwi_list) != len(bvalue_list):
            print('The list of the b values is not consistant to the dwi list')
            return False

        for one_dwi, one_b in zip(dwi_list, bvalue_list):
            if suffex == '.nii':
                store_path = os.path.splitext(dwi_file)[0] + '_b' + one_b + suffex
            elif suffex == '.nii.gz':
                store_path = dwi_file[:-7] + '_b' + one_b + suffex
            nb.save(one_dwi, store_path)
        return True

def ExtractBvalues(candidate_list):
    b_value = []
    for file in candidate_list:
        b_str = ''
        index = -5
        while True:
            if file[index].isdigit():
                b_str = file[index] + b_str
            else:
                b_value.append(int(b_str))
                break
            index -= 1

    return b_value

def FindDWIFile(candidate_list, is_separate=False):
    dwi_list = []
    if is_separate:
        for dwi in candidate_list:
            if (('dwi' in dwi) or ('diff' in dwi)) and ('_b' in dwi) and (('.nii' in dwi) or ('.nii.gz' in dwi)) and \
                    ('Reg' not in dwi) and ('Resize' not in dwi):
                dwi_list.append(dwi)
    else:
        for dwi in candidate_list:
            if (('dwi' in dwi) or ('diff' in dwi)) and ('_b' not in dwi) and (('.nii' in dwi) or ('.nii.gz' in dwi)) and\
                    ('Reg' not in dwi) and ('Resize' not in dwi):
                dwi_list.append(dwi)
    return dwi_list


################################################################################

SAGITTAL = 1  # PFL
CORONAL = 2  # LFP
TRANSVERSE = 3  # LPH

direction_dict = {
    1: 'sagittal',
    2: 'coronal',
    3: 'transverse'
}


class ReformatAxis(object):
    def __init__(self, swap_show=True):
        self._x = np.array([1, 0, 0])  # R->L
        self._y = np.array([0, 1, 0])  # A->P
        self._z = np.array([0, 0, 1])  # F->H
        self._swap_show = swap_show
        self._image = None
        self._InitLog()

    def _InitLog(self):
        self._flip_log = [0, 0, 0]
        self._swap_log = [False, False, False]
        self._axis_direction = [0, 0, 0]
        self._info = ''

    def _AddInfo(self, text):
        self._info += text + '\n'

    def _SwapAxis(self, data, dim1, dim2):
        self._swap_log[dim1] = not self._swap_log[dim1]
        self._swap_log[dim2] = not self._swap_log[dim2]
        self._axis_direction[dim1], self._axis_direction[dim2] = self._axis_direction[dim2], self._axis_direction[dim1]
        return np.swapaxes(data, dim1, dim2)

    def _FlipAxis(self, data, dim):
        self._flip_log[dim] = 1
        self._AddInfo("Flip the direction: {}".format(direction_dict[self._axis_direction[dim]]))
        return np.flip(data, axis=dim)

    def _GetAxis(self, one_direction):
        if (one_direction * self._x).sum().__abs__() >= np.sqrt(0.5):
            return SAGITTAL
        elif (one_direction * self._y).sum().__abs__() >= np.sqrt(0.5):
            return CORONAL
        elif (one_direction * self._z).sum().__abs__() > np.sqrt(0.5):
            return TRANSVERSE
        else:
            print('Error Direction')
            raise NameError

    def Transform(self, image, direction):
        data = sitk.GetArrayFromImage(image)
        data = np.transpose(data)

        if self._axis_direction[2] == SAGITTAL:
            self._AddInfo('Major direction of image is the sagittal direction')
            data = self.FormatSagittal(data, direction)
        elif self._axis_direction[2] == CORONAL:
            self._AddInfo('Major direction of image is the coronal direction')
            data = self.FormatCoronal(data, direction)
        elif self._axis_direction[2] == TRANSVERSE:
            self._AddInfo('Major direction of image is the transverse direction')
            data = self.FormatTransverse(data, direction)
        return data

    def FormatTransverse(self, data, direction):
        # sag, cor, trans
        if self._axis_direction[0] == CORONAL and self._axis_direction[1] == SAGITTAL:
            data = self._SwapAxis(data, 0, 1)
            self._AddInfo('According to LPH, we exchange the x and y')

        if (direction[0] * self._x).sum() < 0:
            data = self._FlipAxis(data, 0)
        if (direction[1] * self._y).sum() < 0:
            data = self._FlipAxis(data, 1)
        if (direction[2] * self._z).sum() < 0:
            data = self._FlipAxis(data, 2)
        return data

    def ViewData(self, data, direction=TRANSVERSE, swap_show=True):
        if swap_show:
            new_data = np.swapaxes(data, 0, 1)
        else:
            new_data = deepcopy(data)

        if isinstance(direction, str) and 'tra' in direction:
            direction = TRANSVERSE
        elif isinstance(direction, str) and 'cor' in direction:
            direction = CORONAL
        elif isinstance(direction, str) and 'sag' in direction:
            direction = SAGITTAL

        if direction == TRANSVERSE:
            if self._axis_direction[2] == SAGITTAL: #PFL-> LPH
                new_data = np.transpose(new_data, (2, 0, 1))
                new_data = np.flip(new_data, axis=2)
            elif self._axis_direction[2] == CORONAL: #LFP -> LPH
                new_data = np.transpose(new_data, (0, 2, 1))
                new_data = np.flip(new_data, axis=2)
        elif direction == SAGITTAL:
            if self._axis_direction[2] == TRANSVERSE: #LPH -> PFL
                new_data = np.transpose(new_data, (1, 2, 0))
                new_data = np.flip(new_data, axis=1)
            elif self._axis_direction[2] == CORONAL: #LFP -> PFL
                new_data = np.transpose(new_data, (2, 1, 0))
        elif direction == CORONAL:
            if self._axis_direction[2] == TRANSVERSE: #LPH -> LFP
                new_data = np.transpose(new_data, (0, 2, 1))
                new_data = np.flip(new_data, axis=1)
            elif self._axis_direction[2] == SAGITTAL: #PFL -> LFP
                new_data = np.transpose(new_data, (2, 1, 0))

        if swap_show:
            new_data = np.swapaxes(new_data, 0, 1)

        return new_data

    def FormatCoronal(self, data, direction):
        # sag, trans, cor
        if self._axis_direction[0] == TRANSVERSE and self._axis_direction[1] == SAGITTAL:
            data = self._SwapAxis(data, 0, 1)
            self._AddInfo('According to LFP, we exchange the x and y')

        if (direction[0] * self._x).sum() < 0:
            data = self._FlipAxis(data, 0)
        if (direction[1] * self._z).sum() > 0:
            data = self._FlipAxis(data, 1)
        if (direction[2] * self._y).sum() > 0:
            data = self._FlipAxis(data, 2)
        return data

    def FormatSagittal(self, data, direction):
        # cor, trans, Sag
        if self._axis_direction[0] == TRANSVERSE and self._axis_direction[1] == CORONAL:
            data = self._SwapAxis(data, 0, 1)
            self._AddInfo('According to PFR, we exchange the x and y')

        if (direction[0] * self._y).sum() < 0:
            data = self._FlipAxis(data, 0)
        if (direction[1] * self._z).sum() > 0:
            data = self._FlipAxis(data, 1)
        if (direction[2] * self._x).sum() > 0:
            data = self._FlipAxis(data, 2)

        return data

    def Run(self, image, is_show_transform_info=False):
        self._image = image
        self._InitLog()

        direction = image.GetDirection()
        direction = np.array([(direction[0], direction[3], direction[6]),
                              (direction[1], direction[4], direction[7]),
                              (direction[2], direction[5], direction[8])])

        self._axis_direction[0] = self._GetAxis(direction[0])
        self._axis_direction[1] = self._GetAxis(direction[1])
        self._axis_direction[2] = self._GetAxis(direction[2])
        assert (np.unique(np.array(self._axis_direction)).size == 3)  # make sure each direction is independent

        data = self.Transform(image, direction)

        # For visualization using python, and this is not logged.
        if self._swap_show:
            data = np.swapaxes(data, 0, 1)
            self._AddInfo("For Imshow3DArray, we exchange the x and y direction")

        if is_show_transform_info:
            print(direction)
            print(self._info[:-1])
        return data

    def BackToImage(self, data):
        if self._image is None:
            print('Run first.')

        if self._swap_show:
            data = np.swapaxes(data, 0, 1)

        for dim, is_flip in enumerate(self._flip_log):
            if is_flip:
                data = np.flip(data, axis=dim)

        swap_dim = tuple(np.where(self._swap_log)[0])
        assert (len(swap_dim) == 2 or len(swap_dim) == 0)
        if len(swap_dim) == 2:
            data = np.swapaxes(data, swap_dim[0], swap_dim[1])

        data = np.transpose(data)

        new_image = sitk.GetImageFromArray(data)
        new_image.CopyInformation(self._image)
        return new_image

#####################################


def BinaryClosing(raw_img, kernel=3, foreground_value=1):
    """

    :param raw_img: sitk image object
    :param kernel: Set the value in the image to consider as "foreground". Defaults to maximum value of PixelType.

    :param foreground_value:  Set/Get the radius of the kernel structuring element as a vector.
                                If the dimension of the image is greater then the length of r, then
                                the radius will be padded. If it is less the r will be truncated.
    :return: sitk image object
    """
    bmc = sitk.BinaryMorphologicalClosingImageFilter()
    bmc.SetKernelType(sitk.sitkBall)
    bmc.SetKernelRadius(kernel)
    bmc.SetForegroundValue(foreground_value)
    bmc_img = bmc.Execute(raw_img)

    return bmc_img


def BinaryOpening(raw_img, kernel=3, foreground_value=1):
    """

    :param raw_img: sitk image object
    :param kernel: Set the value in the image to consider as "foreground". Defaults to maximum value of PixelType.

    :param foreground_value:  Set/Get the radius of the kernel structuring element as a vector.
                                If the dimension of the image is greater then the length of r, then
                                the radius will be padded. If it is less the r will be truncated.
    :return: sitk image object
    """
    bmo = sitk.BinaryMorphologicalOpeningImageFilter()
    bmo.SetKernelType(sitk.sitkBall)
    bmo.SetKernelRadius(kernel)
    bmo.SetForegroundValue(foreground_value)
    bmo_img = bmo.Execute(raw_img)

    return bmo_img


def Norm_min_max(raw_img, minimum=0, maximum=255):

    resacleFilter = sitk.RescaleIntensityImageFilter()
    resacleFilter.SetOutputMaximum(maximum)
    resacleFilter.SetOutputMinimum(minimum)
    image = resacleFilter.Execute(raw_img)
    return image


def NormZ(raw_img, is_show=False):
    normalized_img = sitk.Normalize(raw_img)
    if is_show:
        raw_distribution = sitk.GetArrayFromImage(raw_img).flatten()
        normalized_distribution = sitk.GetArrayFromImage(normalized_img).flatten()

        plt.subplot(121)
        plt.title('raw')
        plt.hist(raw_distribution, bins=40, color='orange')

        plt.subplot(122)
        plt.title('normalization')
        plt.hist(normalized_distribution, bins=40, color='green')

        plt.show()


def HistEq(ref_img, raw_img, is_show=False):

    histeq_img = sitk.HistogramMatching(raw_img, ref_img, numberOfHistogramLevels=660)


    if is_show:
        raw_distribution = sitk.GetArrayFromImage(raw_img).flatten()
        ref_distribution = sitk.GetArrayFromImage(ref_img).flatten()
        histeq_distribution = sitk.GetArrayFromImage(histeq_img).flatten()

        plt.subplot(121)
        plt.title('raw')

        plt.hist(raw_distribution[np.where(raw_distribution>0)],
                 edgecolor='black', bins=40, color='orange')

        plt.subplot(122)
        plt.hist(histeq_distribution[np.where(histeq_distribution>0)], alpha=0.5,
                 edgecolor='black', bins=40, color='green', label='histeq')
        plt.hist(ref_distribution[np.where(ref_distribution>0)], alpha=0.5,
                 edgecolor='black', bins=40, color='red', label='ref')
        plt.legend()
        plt.show()