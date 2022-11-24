from pathlib import Path
import os
dirpath = r'\\219.228.149.7\syli\dataset\Primary and metastatic\zheci\Primary and metastatic OC\Primary and metastatic OC\1-hfz-metastatic-seg（缺）'
dirs = [i for i in Path(dirpath).iterdir()]
for dir in dirs:
    imgs = [i for i in dir.iterdir() if str(i)[-3:] == 'nii']
    rois = [i for i in dir.glob('*nii.gz')]
    for roi in rois:
        new_roi_name = roi.name[-9:]
        # new_roi_name = new_roi_name.insert(2, '_roi')
        old_img_name = str(roi)[:-10]
        if new_roi_name[:2] == 'wi':
            old_img_name = old_img_name[:-1]
            new_roi_name = 'dwi.nii.gz'
        elif new_roi_name[:2] == '+c':
            new_roi_name = 'T1CE.nii.gz'
        else:
            new_roi_name = new_roi_name
        old_img_name = old_img_name + '.nii'
        new_img_name = str(roi.parent) + '/' +new_roi_name[:-3]
        new_roi_name_list = list(new_roi_name)
        new_roi_name_list.insert(-7, '_roi')
        new_roi_name = ''.join(new_roi_name_list)
        new_roi_name = str(roi.parent) + '/' +new_roi_name
        try:
            os.rename(str(roi), new_roi_name)
            os.rename(old_img_name, new_img_name)
        except:
            print(roi)
