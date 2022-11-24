"""
Change the Image return data onto a standard coordinate.

--- All rights reserve.
Yang Song. (songyangmri@gmail.com)

"""

import numpy as np
import SimpleITK as sitk

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
            data = self.TransformToSagittal(data, direction)
        elif self._axis_direction[2] == CORONAL:
            self._AddInfo('Major direction of image is the coronal direction')
            data = self.TransformToCoronal(data, direction)
        elif self._axis_direction[2] == TRANSVERSE:
            self._AddInfo('Major direction of image is the transverse direction')
            data = self.TransformToTransverse(data, direction)
        return data

    def TransformToTransverse(self, data, direction):
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

    def TransformToCoronal(self, data, direction):
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

    def TransformToSagittal(self, data, direction):
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

    def BackToImage(self, data, image=None):
        if image is not None:
            self._image = image

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
        new_image.CopyInformation(image)
        return new_image


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dicom2nifit_list = [r'C:\Users\yangs\Desktop\nii-axis\004_t2_tse_fs_tra.nii.gz',
                        r'C:\Users\yangs\Desktop\nii-axis\003_t2_haste_cor_p3_mbh.nii.gz',
                        r'C:\Users\yangs\Desktop\nii-axis\dicom_direction_test\003_t2_tse_sag_p2_384.nii.gz',
                        r'C:\Users\yangs\Desktop\nii-axis\dicom_direction_test\004_t2_tse_dixon_fs_cor_p2_F.nii.gz',
                        r'C:\Users\yangs\Desktop\nii-axis\dicom_direction_test\006_t2_tse_tra_384_p2.nii.gz',
                        r'C:\Users\yangs\Desktop\nii-axis\dicom_direction_test\009_ep2d_diff_4b2000_tra_p2_ADC.nii.gz']

    dcm2niix_list = [r'C:\Users\yangs\Desktop\nii-axis\004_t2_tse_fs_tra_dcm2niix.nii',
                     r'C:\Users\yangs\Desktop\nii-axis\003_t2_haste_cor_p3_mbh_dcm2niix.nii',
                     r'C:\Users\yangs\Desktop\nii-axis\dicom_direction_test\003_t2_tse_sag_p2_384.nii',
                     r'C:\Users\yangs\Desktop\nii-axis\dicom_direction_test\004_t2_tse_dixon_fs_cor_p2_F.nii',
                     r'C:\Users\yangs\Desktop\nii-axis\dicom_direction_test\006_t2_tse_tra_384_p2.nii',
                     r'C:\Users\yangs\Desktop\nii-axis\dicom_direction_test\009_ep2d_diff_4b2000_tra_p2_ADC.nii']

    ref = ReformatAxis()
    for x_file, t_file in zip(dcm2niix_list, dicom2nifit_list):
        x_image = sitk.ReadImage(x_file)
        x_data = ref.Run(x_image, is_show_transform_info=True)
        x_recover_image = ref.BackToImage(x_data, x_image)

        x_reload = sitk.GetArrayFromImage(x_image)
        x_recover_reload = sitk.GetArrayFromImage(x_recover_image)
        print((x_reload == x_recover_reload).all()) # Assert the recover_image was correct.

        t_image = sitk.ReadImage(t_file)
        t_data = ref.Run(t_image, is_show_transform_info=True)
        t_recover_image = ref.BackToImage(t_data, t_image)

        t_reload = sitk.GetArrayFromImage(t_image)
        t_recover_reload = sitk.GetArrayFromImage(t_recover_image)
        print((t_reload == t_recover_reload).all())  # Assert the recover_image was correct.

        plt.subplot(231)
        plt.imshow(x_data[..., 0], cmap='gray')
        plt.subplot(232)
        plt.imshow(x_data[..., x_data.shape[-1] // 2], cmap='gray')
        plt.subplot(233)
        plt.imshow(x_data[..., -1], cmap='gray')

        plt.subplot(234)
        plt.imshow(t_data[..., 0], cmap='gray')
        plt.subplot(235)
        plt.imshow(t_data[..., t_data.shape[-1] // 2], cmap='gray')
        plt.subplot(236)
        plt.imshow(t_data[..., -1], cmap='gray')
        plt.show()

        print('')
