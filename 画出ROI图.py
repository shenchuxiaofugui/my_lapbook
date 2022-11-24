import numpy as np
import pandas as pd
import SimpleITK as sitk
import os
from pathlib import Path
import matplotlib.pyplot as plt

dirpath = r'\\219.228.149.7\syli\dataset\EC_all\EC_spacing'
savepath = r'\\219.228.149.7\syli\dataset\EC_all\EC_check'


def show_img_label(img_array, roi_array, show_index, title):
    show_img = img_array[show_index, ...]
    show_roi = roi_array[show_index, ...]

    plt.title(title)
    plt.axis('off')
    plt.imshow(show_img, cmap='gray')
    plt.contour(show_roi)
    #plt.show()

def split_roi(roi_arr):
    arrs = []
    item = np.unique(roi_arr)[1:]
    for i in item:
        a = (roi_arr == i)
        arrs.append(a)




def check_img_label(dir_path, savepath, modals):
    case_list = [i for i in Path(dir_path).iterdir()]  #, 'husiyi', 'linjiading', 'maguizhu'

    for case in case_list:
        # filelist = [i for i in Path(case).iterdir() if 'nii' in str(i)]
        # filelist.sort()
        for modal in modals:
            if os.path.exists(savepath + '/' + case.name + f'_{modal}.jpg'):
                continue
            candidate_img = str(case) + f'/{modal}_resampled.nii'
            candidate_roi = str(case) + f'/{modal}_roi_resampled.nii.gz'
            if not os.path.exists(candidate_roi):
                print(case.name)
                continue
            try:
                roi_img = sitk.ReadImage(candidate_roi)
            except:
                print(case.name, "read error")
                continue
            roi_array = sitk.GetArrayFromImage(roi_img) # [slice index, x ,y]
            if np.sum(roi_array) == 0:
                print(candidate_roi)

        #roi_max_index = np.argmax(np.sum(roi_array, axis=(1,2)))
            roi_index = [i for i, e in enumerate(np.sum(roi_array, axis=(1,2)).tolist()) if e != 0]
        #print('ceng', roi_index)
            length = len(roi_index)

            try:
                sub_img = sitk.ReadImage(candidate_img)
            except:
                print(case.name, "read error")
                continue
            img_array = sitk.GetArrayFromImage(sub_img)


            if length > 6:
                roi_index = roi_index[int(length/2)-3:int(length/2)+3]

            #f = plt.subplots(figsize=(40, 40))

            for k in range(len(roi_index)):
                plt.subplot(2, 3, k+1)

                title = modal+str(roi_index[k])
                try:
                    show_img_label(img_array, roi_array, roi_index[k], title)
                except:
                    print(case.name, "huahuashibai")

            plt.savefig(savepath + '/' + case.name +f'_{modal}.jpg', bbox_inches='tight', dpi=400)
            # plt.show()
            plt.close()




def look_direction(dir_path):
    dirs = os.listdir(dir_path)
    for dir in dirs:
        flag = True
        filespath = os.path.join(dir_path, dir)
        for file in os.listdir(filespath):
            if 'label.nii' in file:
                flag = False
                file_path = os.path.join(filespath, file)
                label_direction = get_direction(file_path)
            elif 'BSpline' in file:
                file_path = os.path.join(filespath, file)
                img_direction = get_direction(file_path)
                if img_direction != label_direction:
                    print(dir, ' wrong dirction')
        if flag:
            print(dir, 'wrong')


def get_direction(filepath):
    img = sitk.ReadImage(filepath)
    direction = list(img.GetDirection())
    direction_round = [round(i) for i in direction]
    return direction_round

#os.mkdir(os.path.join(savepath, 'result'))
print('hahahaha')
# check_img_label(dirpath, savepath, ["DWI", "T1CE", "T2"])
roi = sitk.ReadImage(r"\\219.228.149.7\syli\python\nnunet\Infer\Task010_children\301.301_20221005082632818.nii.gz")
arr = sitk.GetArrayFromImage(roi)
split_roi(arr)