"""
MeDIT.Others
Functions that I do not how to label them...

author: Yang Song
All right reserved
"""

import shutil
import math
import sys
import os
import glob

from pathlib import Path
import numpy as np

def IsNumber(string):
    '''
    To adjust the string belongs to a number or not.
    :param string:
    :return:
    '''
    try:
        float(string)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(string)
        return True
    except (TypeError, ValueError):
        pass

    return False

def IsValidNumber(string):
    if not IsNumber(string):
        return False

    if math.isnan(float(string)):
        return False

    return True

def GetPhysicaladdress():
    '''
    @summary: return the MAC address of the computer
    '''

    mac = None
    if sys.platform == "win32":
        for line in os.popen("ipconfig /all"):
            # print line
            if line.lstrip().startswith("Physical Address") or line.lstrip().startswith("物理地址"):
                mac = line.split(":")[1].strip().replace("-", ":")
                break

    else:
        for line in os.popen("/sbin/ifconfig"):
            if 'Ether' in line:
                mac = line.split()[4]
                break
    return mac

def RemoveKeyPathFromPathList(path_list, key_word):
    new_path_list = []
    for p in path_list:
        if key_word not in str(p):
            new_path_list.append(p)

    return new_path_list

def CopyFile(source_path, dest_path, is_replace=True):
    source_path = str(source_path)
    dest_path = str(dest_path)
    if not os.path.exists(source_path):
        print('File does not exist: ', source_path)
        return None
    if not (os.path.exists(dest_path) and (not is_replace)):
        shutil.copyfile(source_path, dest_path)

def CopyFiles(source_path_list, dest_folder, is_replace=True, new_name=''):
    if isinstance(source_path_list, Path):
        source_path_list = str(source_path_list)
    if isinstance(source_path_list, str):
        source_path_list = [source_path_list]

    if not os.path.isdir(dest_folder):
        print('The Folder does not exist: {}'.format(dest_folder))
        return None

    for source_path in source_path_list:
        if new_name != '':
            dest_path = os.path.join(dest_folder, new_name + os.path.splitext(source_path)[-1])
        else:
            dest_path = os.path.join(dest_folder, os.path.split(source_path)[1])
        CopyFile(source_path, dest_path, is_replace)

def CopyFolder(source_folder, dest_folder, is_replace=True):
    if not os.path.isdir(source_folder):
        print('Folder does not exist: ', source_folder)
        return None
    if (not os.path.exists(dest_folder)) or is_replace:
        if os.path.isdir(dest_folder):
            shutil.rmtree(dest_folder)
        shutil.copytree(source_folder, dest_folder)

def MoveFile(source_path, dest_path, is_replace):
    source_path = str(source_path)
    dest_path = str(dest_path)
    if not os.path.exists(source_path):
        print('File does not exist: ', source_path)
        return None
    if (not os.path.exists(dest_path)) or is_replace:
        shutil.move(source_path, dest_path)

def MoveFiles(source_path_list, dest_folder, is_replace=True, new_name=''):
    if isinstance(source_path_list, Path):
        source_path_list = str(source_path_list)
    if isinstance(source_path_list, str):
        source_path_list = [source_path_list]

    if not os.path.isdir(dest_folder):
        print('The Folder does not exist: {}'.format(dest_folder))
        return None

    for source_path in source_path_list:
        if new_name != '':
            dest_path = os.path.join(dest_folder, new_name + os.path.splitext(source_path)[-1])
        else:
            dest_path = os.path.join(dest_folder, os.path.split(source_path)[1])
        MoveFile(source_path, dest_path, is_replace)

def MakeFolder(folder_path):
    try:
        if isinstance(folder_path, str):
            folder_path = Path(folder_path)
        if not folder_path.exists():
            folder_path.mkdir()
    except FileNotFoundError:
        folder_path = str(folder_path)
        os.makedirs(folder_path)
        folder_path = Path(folder_path)
    return folder_path

##################################################

def HumanSortFile(file_list):
    import re

    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    file_list.sort(key=alphanum_key)
    return file_list

def SplitPathWithSuffex(file_path):
    if file_path.endswith('.nii.gz'):
        return file_path[:-len('.nii.gz')], '.nii.gz'
    else:
        return os.path.splitext(file_path)

def CompareSimilarityOfLists(*input_lists):
    if len(input_lists) < 2:
        print('At least 2 lists')

    max_diff = -1.
    for one in input_lists:
        for second in input_lists[input_lists.index(one) + 1:]:
            one_array = np.asarray(one)
            second_array = np.asarray(second)
            diff = np.sqrt(np.sum(np.square(one_array - second_array)))
            if diff > max_diff:
                max_diff = diff

    return max_diff

def IterateCase(root_folder_path, verbose=True, start_from='', only_folder=True):
    if isinstance(root_folder_path, str):
        root_folder_path = Path(root_folder_path)

    for case in sorted(root_folder_path.iterdir()):
        if start_from != '' and case.name.lower() < start_from.lower():
            continue
        if case.name == '$RECYCLE.BIN' or case.name == 'System Volume Information':
            continue
        if only_folder and (not case.is_dir()):
            continue
        if verbose:
            print(case.name)
        yield case

def AddNameInEnd(file_path, name):
    file_name, suffext = SplitPathWithSuffex(str(file_path))
    return file_name + '_' + name + suffext

def GlobNumber(pattern, max_length=10):
    if '[0-9]' not in pattern:
        print('There is no number in the pattern')
        return pattern

    candidate = []
    subs = pattern.split('[0-9]')
    for index in range(max_length):
        p = ('[0-9]' * (index + 1)).join(subs)
        candidate.extend(glob.glob(p))

    return candidate


def FileSplitext(file_path):
    if file_path.endswith('.nii.gz'):
        return file_path[:-len('.nii.gz')], '.nii.gz'
    else:
        return os.path.splitext(file_path)


if __name__ == '__main__':
    print(FileSplitext(r'u:\JSPH\ProstateCancer\2020Supplement\BI '
                       r'JUN\MR\20190608\234415\18382\007_epi_dwi_tra_CBSF5_NL3_b50_merge.nii'))
