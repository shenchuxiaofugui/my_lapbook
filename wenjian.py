import os
import pandas as pd
from xpinyin import Pinyin
import SimpleITK as sitk
import shutil


def copy(filepath, target):
    #复制文件
    file = os.path.basename(filepath)
    with open(filepath, 'rb') as rstream:
        container = rstream.read()
        path1 = os.path.join(target,file)
        with open(path1, 'wb') as wstream:
            wstream.write(container)


def guina(filepath, new_path):
    #每个病人每个模态的原文件和roi文件放在一个文件夹，且第一个.前面是他们的名字，将他们整理到各自的文件夹中
    file_list = os.listdir(filepath)
    file_list.sort()
    case_name = file_list[0]
    case_name = case_name[:case_name.index('.')]
    new_file = os.path.join(new_path, case_name)
    os.mkdir(new_file)
    for img in file_list:
        name = img[:img.index('.')]
        if name == case_name:
            copy(os.path.join(filepath, img), new_file)
        else:
            case_name = img[:img.index('.')]
            new_file = os.path.join(new_path, case_name)
            os.mkdir(new_file)
            copy(os.path.join(filepath, img), new_file)


def rename(filepath, modals):
    #已经一个病人一个文件夹，将文件夹里面的名字重命名（前提是原名字中包含模态，且roi文件是gz压缩文件），modals是模态的列表参数
    # name = name[name.rfind('.', -1) + 1:]
    filename = os.path.basename(filepath)
    filename = filename.upper()
    #filename = filename.replace('TI', 'T1')
    filename = filename.replace('T1+', 'T1CE')
    if filename[-3:] == 'NII':
        houzhui = '.nii'
    else:
        houzhui = '_roi.nii.gz'
    new_path = os.path.dirname(filepath)
    for modal in modals:
        if modal in filename:
            if modal == 'T1':
                if 'T1CE' not in filename:
                    new_name = os.path.join(new_path, 'T1') + houzhui
            elif modal == 'SAG':
                new_name = os.path.join(new_path, 'T2_SAG') + houzhui
            elif modal == '+C':
                new_name = os.path.join(new_path, 'T1CE') + houzhui
            else:
                new_name = os.path.join(new_path, modal) + houzhui
            try:
                os.rename(filepath, new_name)
            except (Exception, BaseException) as e:
                print(filepath)
                print(e)


def batch_rename(filepath, modals):
    #批量执行nii（.gz)文件重命名操作
    files = os.listdir(filepath)
    for file1 in files:
        path = os.path.join(filepath, file1)
        dirs = os.listdir(path)
        for file in dirs:
            rename(os.path.join(path, file), modals)


def check(filepath, modals):
    #检查所有文件是否齐全，一一对应，number是文件夹里面应该有的文件数，length是文件名长度应该小于length，check_none检查有没有空的文件
    files = os.listdir(filepath)
    all_file = []
    for modal in modals:
        all_file.append(modal + '.nii')
        all_file.append(modal + '_roi.nii')
    all_file.sort()
    for file1 in files:
        path = os.path.join(filepath, file1)
        dirs = os.listdir(path)
        if dirs != all_file:
            print(file1, len(dirs))
            print(dirs)


def change_case_name(filepath, line='Name', save=False):
    #将病人信息表的中文名字改为拼音，line是中文名对应的列名，并返回拼音与ID的对应表（前提是ID在第一列）
    df = pd.read_excel(filepath)
    p = Pinyin()
    case_name = []
    for i in df[line]:
        case_name.append(p.get_pinyin(i, '_').upper())
    df.insert(df.columns.get_loc('Name') + 1, 'pinyin', case_name)
    if save:
        df.to_csv(filepath, index=False, encoding="utf_8_sig")
    ids = df.set_index('pinyin').iloc[:, 0]
    return ids


def change_file_name(filepath, df):
    #将病人的文件夹名由拼音改到ID
    files = os.listdir(filepath)
    ids = change_case_name(df)
    #os.rename(os.path.join(new_path, dir), os.path.join(new_path, ''.join(list(filter(str.isdigit, dir)))))
    for file in files:
        try:
            new_name = os.path.join(filepath, ids[file])
            os.rename(os.path.join(filepath, file), new_name)
        except (Exception, BaseException) as e:
            print(file)
            print(e)


def rename_2(filepath):
    #针对roi文件没有写模态，批量改名
    files = os.listdir(filepath)
    for file1 in files:
        path = os.path.join(filepath, file1)
        if os.path.isdir(path):
            dirs = os.listdir(path)
            for file in dirs:
                if file[-3:] == 'nii':
                    for casename in dirs:
                        if file[:-4] == casename[:casename.rfind('.', 0, -3)] and casename[-6:] == 'nii.gz':
                            new_name = casename[len(file[:-3]):-7]
                            new_name = new_name.upper() + '.nii'
                            old_name = os.path.join(path, file)
                            new_name = os.path.join(path, new_name)
                            try:
                                os.rename(old_name, new_name)
                            except (Exception, BaseException) as e:
                                print(file1)
                                print(e)


def rename_3(filepath, modals):
    #针对roi文件没有写模态，批量改名
    files = os.listdir(filepath)
    for file1 in files:
        path = os.path.join(filepath, file1)
        if os.path.isdir(path):
            dirs = os.listdir(path)
            for file in dirs:
                filename = file.upper()
                for modal in modals:
                    if modal in filename:
                        if modal == 'T1':
                            if 'T1+' not in filename:
                                new_name = os.path.join(path, 'T1')
                        # elif modal == 'SAG':
                        #     new_name = os.path.join(new_path, 'T2_SAG') + houzhui
                        elif modal == '+C':
                            new_name = os.path.join(path, 'T1CE')
                        else:
                            new_name = os.path.join(path, modal)
                        if filename[-3:] == 'NII':
                            new_name = new_name + '.nii'
                        else:
                            new_name = new_name + '_roi.nii.gz'
                        try:
                            os.rename(os.path.join(path, file), new_name)
                        except (Exception, BaseException) as e:
                            print(path)
                            print(e)




def check_roi(filepath):
    dirs = os.listdir(filepath)
    for dir in dirs:
    #for dir in ['5083585', '5379464', '5469332']:
        path = os.path.join(filepath, dir)
        files = os.listdir(path)
        for file in files:
            if file[-3:] == 'nii':
                case_name = os.path.join(path, file)
                roi_name = os.path.join(path, file[:-4]+'_roi.nii.gz')
                try:
                    case_img = sitk.ReadImage(case_name)
                    roi_img = sitk.ReadImage(roi_name)
                    case_arr = sitk.GetArrayFromImage(case_img)
                    roi_arr = sitk.GetArrayFromImage(roi_img)
                    if roi_arr.sum() == 0:
                        print(dir, file, 'no data')
                        continue
                    if case_arr.shape != roi_arr.shape:
                        print(dir, file, 'no match')
                        continue
                    roi_img.CopyInformation(case_img)
                    sitk.WriteImage(roi_img, roi_name)
                except (Exception, BaseException) as e:
                    print(dir, file)
                    print(e)


def check_dirs(df_path, key, dirpath):
    dirs = os.listdir(dirpath).sort()
    df = pd.read_csv(df_path)[key].tolist().sort()
    if dirs != df:
        inter = list(set(dirs).intersection(set(df)))
        print(dirs - inter)
        print(df - inter)
    else:
        print('yes yes yes')






path = r'\\mega\syli\dataset\EC_seg\EC-old'     #所有文件在一个文件夹，那个文件夹的地址
new_path = r'\\219.228.149.7\syli\dataset\Primary and metastatic\zheci\Primary and metastatic OC\Primary and metastatic OC\1-hfz-metastatic-seg（缺）'   #要整理到新的文件夹的地址
df_path = r'\\mega\syli\dataset\EC_seg\process_old_EC_seg(WTP)\process_old_EC1.csv'     #病人信息表的地址（ID在第一列）
modals = ['+C', 'DWI', 'T2', 'T1']   #这次有哪些模态
#guina(path, new_path)    #整理文件到新地址
#change_file_name(new_path, df_path)     #修改文件夹的名字
rename_3(new_path, modals)        #针对nii文件名字没有模态信息，但是roi文件名由模态信息，且roi文件模态前的字符与原图一一对应
#batch_rename(new_path, modals)        #批量修改每个文件的名字（roi得是gz结尾）
#check(new_path, ['T1CE', 'DWI', 'T2'])       #第二个参数是文件里应该有的文件数,check_none检查有没有空的文件
#check_roi(new_path)
#check_dirs(df_path, '影像号', new_path)



