    # -*- coding: utf-8 -*-
import os
from MeDIT.Visualization import Imshow3DArray
from MeDIT.SaveAndLoad import LoadNiiData
import numpy as np

# excel_path = r"Z:\WHJ\LYM\宫颈癌信息-处理后.xlsx"
# data = pd.read_excel(excel_path, sheet_name='汇总', header=0)

cases_path = r'E:\yidong\ly'
modals = ['ADC', 'T1CE', 'T1', 'T2']
cases = os.listdir(cases_path)
cases.sort(reverse=False)
#cases = cases[cases.index('5641783'):]
#cases = cases[80:]
case_number = len(cases)
i = 0
print(case_number)

for case in cases[0:]:
    print('case:', case, '剩余:', case_number-i)
    i = i+1
    for modal in modals:
        images_path = os.path.join(cases_path, case, f'{modal}.nii')
        # images = os.listdir(images_path)
        roi_path = os.path.join(cases_path, case, f'{modal}_roi.nii')
        #_, nii_data, _ = LoadNiiData(images_path)
        try:
            _, nii_data, _ = LoadNiiData(images_path)
        except:
            print('hahhhaha')
            continue
        _, roi_data, _ = LoadNiiData(roi_path)
        #nii_data = np.clip(nii_data, 0, np.percentile(nii_data, 99.95))
        #nii_data = (nii_data-np.min(nii_data))/(np.max(nii_data) - np.min(nii_data))
        Imshow3DArray(nii_data, roi_data)
    # for image in images:
    #     if 'Scene' in image or 'Segmentation' in image:
    #         continue
    #     if 'T2FLAIR' not in image:
    #         continue
    #     image_path = os.path.join(images_path, image)
    #     _, nii_data, _ = LoadNiiData(image_path)
    #     nii_data = (nii_data - np.min(nii_data)) / (np.max(nii_data) - np.min(nii_data))
    #     roi_1[roi_1 != 1] = 0
    #     print(roi_1.sum())
    #     roi_2 = roi_data.copy()
    #     roi_2[roi_2 != 2] = 0
    #     print(roi_2.sum()/2)
    #     roi_3 = roi_data.copy()
    #     roi_3[roi_3 != 3] = 0
    #     print(roi_3.sum()/3)
    #     rois = [roi_1, roi_2, roi_3]
    #
    #     for roi in range(3):
    #         print('image:', image, 'roi:', roi+1)
    #         Imshow3DArray(nii_data, rois[roi]/(roi+1))
    #
    #     # print(nii_data.shape)
    #     # print(roi0_data.shape)
    #     # print(roi1_data.shape)
    #     print('\n')
        print('case:', case, 'modal:', modal, 'min voxel:', nii_data.min(), 'max voxel:', nii_data.max())
