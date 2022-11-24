import pydicom
import os
import csv
import argparse
from pathlib import Path
import pandas as pd

KEYS = ['PatientID', 'SeriesUID', 'Age', 'Sex', 'Path', 'StudyDate', 'Modal', 'Plane', 'SeriesID', 'StudyID', 'department', 'manufacturer', 'institution', 'station', 'magnetic']

def GetDicomModal(caption):
    caption = caption.upper()
    if caption.find("T1") != -1:
        return "T1"
    if caption.find("T2") != -1:
        return "T2"
    return "Unknown"

def GetMainDir(dir):
    x = dir[0]
    y = dir[1]
    z = dir[2]
    if abs(x) > abs(y) and abs(x) > abs(z):
        return [1, 0, 0]
    if abs(y) > abs(x) and abs(y) > abs(z):
        return [0, 1, 0]
    if abs(z) > abs(x) and abs(z) > abs(y):
        return [0, 0, 1]

def GetPlane(orient_list):
    row_dir = orient_list[0:3]
    col_dir = orient_list[3:]
    plane_dic = {"ax": [1, 0, 0, 0, 1, 0], "sag": [0, 1, 0, 0, 0, 1], "cor": [1, 0, 0, 0, 0, 1]}
    dir = GetMainDir(row_dir) + GetMainDir(col_dir)
    plane = "unknown"
    for k,v in zip(plane_dic.keys(), plane_dic.values()):
        if dir == v:
            plane = k
    return plane

def IsDcmFile(file_path):
    if file_path.find("DICOMDIR") != -1:
        return False
    try:
        dcm = pydicom.dcmread(file_path)
        data = dcm[0x20, 0x0E]
        return True
    except Exception:
        return False

def GetPatient(path, index):
    try:
        res = path.split('\\')
        return res[index]
    except Exception as e:
        print("Error Msg:", e)
        return path

def GetDicomeInfo(file_path, header, exist_series):
    info = {}
    # file_path = r'F:\\data\\MR\\wuguan_whang\\ki67\\50up\\A SCC\\A02 p00364851\\DWI\\20RWQCKB\\UBIE4WOV\\I4200000'
    try:
        dcm = pydicom.dcmread(file_path)
        if dcm.PatientID in exist_series:
            return info
        else:
            for key in header:
                if key not in KEYS:
                    print(f'{key} is not available!')
                    continue
                if key == "PatientID":
                    info["PatientID"] = dcm.PatientID
                elif key == "SeriesUID":
                    info["SeriesUID"] = dcm[0x20, 0x0E].value
                elif key == 'Age':
                    info["Age"] = dcm[0x10, 0x1010].value[1:3]
                elif key == 'Sex':
                    info["Sex"] = dcm[0x10, 0x0040].value
                elif key == 'Path':
                    info["Path"] = GetPatient(file_path, 7)
                elif key == 'StudyDate':
                    info["StudyDate"] = dcm.StudyDate
                elif key == 'Modal':
                    caption = dcm[0x08, 0x103E].value
                    info["Modal"] = GetDicomModal(caption)
                elif key == 'Plane':
                    orient_list = dcm[0x20, 0x37].value
                    info["Plane"] = GetPlane(orient_list)
                elif key == 'SeriesID':
                    info["SeriesID"] = dcm[0x20, 0x11].value
                elif key == 'StudyID':
                    info['StudyID'] = dcm[0x20,0x10].value
                elif key == 'department':
                    info['department'] = dcm[0x08,0x1040].value
                elif key == 'manufacturer':
                    info['manufacturer'] = dcm[0x08,0x70].value
                elif key == 'institution':
                    info['institution'] = dcm[0x08,0x80].value
                elif key == 'station':
                    info['station'] = dcm[0x08,0x1010].value
                elif key == 'magnetic':
                    info['magnetic'] = dcm[0x18, 0x87].value
            exist_series.append(dcm.PatientID)

            return info
    except Exception as e:
        print("Error Msg:", e)
        return {}

def SaveInfo(info_list, csv_header, store_path):
    with open(store_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(csv_header)
        for info in info_list:
            writer.writerow(info)

if __name__ == '__main__':
    # 待扫描的根目录, 本脚本会递归搜索所有子目录
    filepath = r'\\219.228.149.7\syli\dataset\zj_data\jly\data'
    all_info = []
    # flag = 0
    for i in Path(filepath).iterdir():
        for j in i.iterdir():
            casedir = str(j)
            # if 'hongsong' in casedir:
            #     flag = 1
            if os.path.isdir(casedir):
            # 想要输出的信息
                subdirs = [k for k in Path(casedir).glob("*T2*")]
                #assert len(subdirs) == 1, "长度不对"
                subdir = subdirs[0]
                csv_header = ['PatientID', 'magnetic']
                info_list = []
                exist_ids = []
                for f in os.listdir(subdir):
                    file = os.path.join(subdir, f)
                    if os.path.isfile(file):
                        print('Found dicom in dir:', subdir)
                        info = GetDicomeInfo(file, csv_header, exist_ids)
                        if len(info) != 0:
                            info_list.append(info.values())
                out_dir = str(i)
                all_info.append(info_list)
                SaveInfo(info_list, csv_header, os.path.join(out_dir, 'dicom_magnetic_info.csv'))
                break
        # 输出csv的目录
    a = pd.DataFrame(all_info)
    print(a)