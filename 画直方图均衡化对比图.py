import numpy as np
import pandas as pd
import SimpleITK as sitk
import os
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import exposure

dirpath = r'\\219.228.149.7\syli\dataset\Primary and metastatic\zheci\yfy Primary and metastatic-seg\ALL'
savepath = r'\\219.228.149.7\syli\dataset\Primary and metastatic\zheci\yfy Primary and metastatic-seg\CLAHE'


def show_img_label(img_array, roi_array, show_index, title):
    show_img = img_array[show_index, ...]
    show_roi = roi_array[show_index, ...]

    plt.title(title)
    plt.imshow(show_img, cmap='gray')
    plt.contour(show_roi)
    #plt.show()



def check_img_label(dir_path, modals):
    case_list = [i for i in Path(dirpath).iterdir()]  #, 'husiyi', 'linjiading', 'maguizhu'

    for case in case_list:
        for modal in modals:
        # filelist = [i for i in Path(case).iterdir() if 'nii' in str(i)]
        # filelist.sort()
        # if os.path.exists(savepath + '/' + case.name +'.jpg'):
        #     continue
            candidate_img = str(case) + f'/{modal}.nii'
            candidate_roi = str(case) + f'/{modal}_roi.nii.gz'
            if not os.path.exists(candidate_roi):
                print(case.name, 'no roi')
                continue
            try:
                roi_img = sitk.ReadImage(candidate_roi)
            except:
                print(case.name, "read error")
                continue
            roi_array = sitk.GetArrayFromImage(roi_img) # [slice index, x ,y]
            if np.sum(roi_array) == 0:
                print("kong", candidate_roi)

            roi_max_index = np.argmax(np.sum(roi_array, axis=(1,2)))
        #roi_index = [i for i, e in enumerate(np.sum(roi_array, axis=(1,2)).tolist()) if e != 0]
        #length = len(roi_index)

            try:
                sub_img = sitk.ReadImage(candidate_img)
            except:
                print(case.name, "read error")
                continue
            img_array = sitk.GetArrayFromImage(sub_img)


            # if length > 8:
            #     roi_index = roi_index[int(length/2)-4:int(length/2)+4]

        #f = plt.subplots(figsize=(40, 40))



            plt.subplot(1, 2, 1)

            title1 = modal + '_original'
            try:
                show_img_label(img_array, roi_array, roi_max_index, title1)
            except:
                print(case.name, "huahuashibai")
            plt.subplot(1, 2, 2)
            img_arr = np.clip(img_array, 0, np.percentile(img_array, 99.95))
            img_arr = (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))
            img_arr = exposure.equalize_adapthist(img_arr)
            title2 = modal + '_CLAHE'
            show_img_label(img_arr, roi_array, roi_max_index, title2)
            if not os.path.exists(savepath + f'/{modal}'):
                os.mkdir(savepath + f'/{modal}')
            plt.savefig(savepath + f'/{modal}/' + case.name +'.jpg')
            plt.show()
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
check_img_label(dirpath,['dwi', 'T1CE', 't1', 't2'])
