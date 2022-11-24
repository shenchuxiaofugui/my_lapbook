'''
MeDIT.SaveAndLoad
Functions for sampling the k-space.

author: Yang Song
All right reserved
'''

import pickle
import numpy as np
import h5py
import nibabel as nb
import os
import cv2
import SimpleITK as sitk
import pydicom
from pathlib import Path
import imageio
import pandas as pd
from copy import deepcopy
from collections import OrderedDict
import matplotlib.pyplot as plt

from MeDIT.ImageProcess import GetImageFromArrayByImage, GetDataFromSimpleITK, ReformatAxis
from MeDIT.Normalize import IntensityTransfer


def SaveDict(one_dict, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(one_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def LoadDict(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)


def LoadCSVwithChineseInPandas(file_path, **kwargs):
    if 'encoding' not in kwargs.keys():
        return pd.read_csv(file_path, encoding="gbk", **kwargs)
    else:
        return pd.read_csv(file_path, **kwargs)

def SaveH5(store_path, data, tag, data_type=np.float32):
    if not isinstance(data, list):
        data = [data]
    if not isinstance(tag, list):
        tag = [tag]
    if not isinstance(data_type, list):
        data_type = [data_type for index in range(len(data))]

    file = h5py.File(str(store_path), 'w')
    for index in range(len(data)):
        file.create_dataset(tag[index], data=data[index], dtype=data_type[index])
    file.close()

def LoadH5(data_path, tag, data_type=np.float32):
    with h5py.File(str(data_path), 'r') as file:
        if not isinstance(tag, list):
            tag = [tag]
        assert(len(tag) >= 1)
        if not isinstance(data_type, list):
            data_type = [data_type for index in range(len(tag))]

        if len(tag) == 1:
            data = np.asarray(file[tag[0]], dtype=data_type[0])
        elif len(tag) > 1:
            data = [np.asarray(file[one_tag], dtype=one_data_type) for one_tag, one_data_type in zip(tag, data_type)]

    return data

def LoadH5AllTag(data_path):
    with h5py.File(data_path, 'r') as file:
        keys = deepcopy(list(file.keys()))
    return keys, LoadH5(data_path, tag=keys)

def SaveArrayAsImage(array, store_path, dpi=(300, 300), format='TIFF', size=5):
    import PIL.Image
    save_array = np.asarray(IntensityTransfer(array, 255, 0), dtype=np.uint8)

    arr2im = PIL.Image.fromarray(save_array)
    if isinstance(size, int):
        size = (size, int(size / save_array.shape[1] * save_array.shape[0]))

    if len(size) == 2:
        new_arry = arr2im.resize([dpi[t] * size[t] for t in range(2)], PIL.Image.NEAREST)
        new_arry.save(store_path, format=format, dpi=dpi)
    elif len(size) == 0:
        arr2im.save(store_path, format=format, dpi=dpi)

def LoadH5InfoForGenerate(data_path):
    file = h5py.File(data_path, 'r')
    info = {'input_number': 0, 'output_number': 0}
    key_list = []
    for key in file.keys():
        key_list.append(key)
        input_output, current_number = key.split('_')
        if input_output == 'input':
            info['input_number'] = np.max([int(current_number) + 1, info['input_number']])
        elif input_output == 'output':
            info['output_number'] = np.max([int(current_number) + 1, info['output_number']])
        else:
            print('Error:', data_path)

    return info

def LoadKerasWeightH5Info(data_path):
    '''
    Load the h5file and print all the weights.
    :param data_path: the path of the h5 file.
    :return:
    '''
    data_dict = h5py.File(data_path, 'r')
    for top_group_name in data_dict.keys():
        print(top_group_name)
        for group_name in data_dict[top_group_name].keys():
            print('    ' + group_name)
            for data_name in data_dict[top_group_name][group_name].keys():
                print('        ' + data_name + ': ' + str(np.shape(data_dict[top_group_name][group_name][data_name].value)))

def LoadNiiData(file_path, dtype=np.float32, is_show_info=False, is_flip=True, flip_log=[0, 0, 0]):
    file_path = str(file_path)

    image = sitk.ReadImage(file_path)
    data, show_data = GetDataFromSimpleITK(image, dtype=dtype)

    if is_show_info:
        print('Image size is: ', image.GetSize())
        print('Image resolution is: ', image.GetSpacing())
        print('Image direction is: ', image.GetDirection())
        print('Image Origion is: ', image.GetOrigin())

    return image, data, show_data

def LoadNiiDcm2Niix(file_path, dtype=np.float32, is_show_info=False):
    # 使用dcm2niix.exe （201904版本）转换的nii，其方向在1、2两个轴上会反向。
    if isinstance(file_path, Path):
        file_path = str(file_path)

    image = sitk.ReadImage(file_path)
    data = np.asarray(sitk.GetArrayFromImage(image), dtype=dtype)

    show_data = np.transpose(deepcopy(data))
    show_data = np.flip(show_data, axis=(1, 2))
    show_data = np.swapaxes(show_data, 0, 1)

    if is_show_info:
        print('Image size is: ', image.GetSize())
        print('Image resolution is: ', image.GetSpacing())
        print('Image direction is: ', image.GetDirection())
        print('Image Origion is: ', image.GetOrigin())

    return image, data, show_data

def LoadNiiHeader(file_path, is_show_info=True):
    if isinstance(file_path, Path):
        file_path = str(file_path)

    reader = sitk.ImageFileReader()
    reader.SetFileName(file_path)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    info = OrderedDict()
    for k in reader.GetMetaDataKeys():
        v = reader.GetMetaData(k)
        info[k] = v

    dimension = int(info['dim[0]'])
    spaceing = []
    center = [float(info['qoffset_x']), float(info['qoffset_y']), float(info['qoffset_z'])]
    shape = []
    for d in range(dimension):
        spaceing.append(float(info['pixdim[' + str(d+1) + ']']))
        shape.append(int(info['dim[' + str(d + 1) + ']']))

    info['dimension'] = dimension
    info['spacing'] = spaceing
    info['center'] = center
    info['shape'] = shape

    if is_show_info:
        print(info)

    return info

def LoadImage(file_path, dtype=np.float32, is_show_info=False):
    file_path = str(file_path)
    image = sitk.ReadImage(file_path)

    ref = ReformatAxis()
    data = ref.Run(image)

    if is_show_info:
        print('Image size is: ', image.GetSize())
        print('Image resolution is: ', image.GetSpacing())
        print('Image direction is: ', image.GetDirection())
        print('Image Origion is: ', image.GetOrigin())

    return image, data.astype(dtype), ref

def SaveNiiImage(store_path, image):
    if isinstance(store_path, Path):
        store_path = str(store_path)
    sitk.WriteImage(image, store_path)

def SaveArrayToNiiByRef(store_path, array, ref_image):
    if isinstance(store_path, Path):
        store_path = str(store_path)

    image = GetImageFromArrayByImage(array, ref_image)
    image.CopyInformation(ref_image)
    sitk.WriteImage(image, store_path)

def SaveDicomByRefer(store_path, data, dicom_data):
    if isinstance(store_path, Path):
        store_path = str(store_path)

    if isinstance(dicom_data, str) and dicom_data[-3:] == 'dcm':
        dicom_data = pydicom.dcmread(dicom_data)

    try:
        assert(data.shape == dicom_data.pixel_array.shape)
    except:
        print("The data shape must be same with the shape of the dicom data. ")

    ds = deepcopy(dicom_data)
    ds.PixelData = data.tostring()
    ds.save_as(store_path)

def SaveSiemens2DDicomSeries(array, dicom_folder, store_folder):
    # Sort the array according the SliceLocation
    dicom_file_list = os.listdir(dicom_folder)
    dicom_file_list.sort()

    slice_location_list = []
    for dicom_file in dicom_file_list:
        dicom_data = pydicom.dcmread(os.path.join(dicom_folder, dicom_file))
        slice_location_list.append(float(dicom_data.SliceLocation))

    sort_index_list = sorted(range(len(slice_location_list)), key=lambda k: slice_location_list[k])
    sort_index_list = sorted(range(len(sort_index_list)), key=lambda k: sort_index_list[k])

    for dicom_file, store_index in zip(dicom_file_list, range(len(dicom_file_list))):
        dicom_data = pydicom.dcmread(os.path.join(dicom_folder, dicom_file))

        ds = deepcopy(dicom_data)
        ds.PixelData = array[..., sort_index_list[store_index]].tostring()
        ds.save_as(os.path.join(store_folder, str(store_index) + '.dcm'))

def LoadDicomData(data_path):
    ds = pydicom.dcmread(data_path)
    data = ds.pixel_array

    return ds, data

def SaveAsGif(store_path, data_list, duration=1):
    gif = []
    for image in data_list:
        gif.append(np.asarray(deepcopy(image), dtype=np.uint8))

    imageio.mimsave(store_path, gif, duration=duration)

def LoadFreeSurferLabel(ref_image_path, label_file, store_folder=''):
    if store_folder:
        if not os.path.exists(store_folder):
            os.mkdir(store_folder)

    def _GetLabel(label_path):
        with open(label_path, 'r') as file:
            lines = file.readlines()
            position_list = []
            for line in lines[2:]:
                one_line = line.split(' ')
                one_line = [index for index in one_line if index != '']
                position_list.append([float(one_line[1]), float(one_line[2]), float(one_line[3])])

        return position_list

    if not (isinstance(ref_image_path, str) and isinstance(label_file, str)):
        print('The input must be file path (string)')
    flip_log = [0, 0, 0]
    if ref_image_path.endswith('.mgz'):
        temp_image = nb.load(ref_image_path)
        if store_folder:
            nb.save(temp_image, os.path.join(store_folder, 'transfer_image.nii.gz'))
            image, data, show_data = LoadNiiData(os.path.join(store_folder, 'transfer_image.nii.gz'), is_flip=flip_log)
        else:
            temp_image_path = 'temp_image.nii.gz'
            nb.save(temp_image, temp_image_path)
            # image, data, show_data = LoadNiiData(temp_image_path, is_flip=flip_log)
            image, show_data, ref = LoadImage(temp_image_path)
            os.remove(temp_image_path)

    if ref_image_path.endswith('.nii') or ref_image_path.endswith('.nii.gz'):
        # image, data, show_data = LoadNiiData(ref_image_path, is_flip=flip_log)
        image, show_data, ref = LoadImage(temp_image_path)

    origion = image.GetOrigin()
    image.SetOrigin((128, 128, 128))
    position_list = _GetLabel(label_file)
    index_list = list(set([image.TransformPhysicalPointToIndex(position) for position in position_list]))
    mask = np.zeros((image.GetSize()), dtype=np.uint8)
    for index in index_list:
        mask[index[1], index[0], index[2]] = 1

    if store_folder:
        from MeDIT.ImageProcess import GetImageFromArrayByImage
        image.SetOrigin(origion)
        store_image = GetImageFromArrayByImage(mask, image, is_transfer_axis=True)
        SaveNiiImage(os.path.join(store_folder, 'label.nii.gz'), store_image)

    return show_data, mask

def MPRoitoAIStationRoi(input_path, output_path):
    img = np.load(input_path)

    ca_index = 1;
    roi_index = 1;

    print(img.keys())
    roi_labels = img['ROILabel'];
    roi_tag = img['ROILabelTag'];
    label_img = img['ROIdataArray']
    points = img['ROIpointArray'];

    if (img['ROILabel'].shape[0] != label_img.shape[0]) or (img['ROILabel'].shape[0] != points.shape[0]):
        exit(1)

    print(points.shape)

    roi_file = station_roi_pb2.RoiLabelFile()

    for index in range(roi_labels.shape[0]):
        roi = roi_file.labels.add()
        image_finding = roi.attribute.add();
        image_finding.category = 'image_finding'
        # print(points.shape)
        # print(type(points[index]))

        slice = len(points[index])
        # print('slice', slice)
        for att_index in range(roi_tag.size):
            if roi_tag[att_index] == 'T2':
                image_finding.attribute['T2_PIRADS'] = roi_labels[index, att_index]
            elif roi_tag[att_index] == 'DWI':
                image_finding.attribute['DWI_PIRADS'] = roi_labels[index, att_index]
            elif roi_tag[att_index] == 'DCE':
                image_finding.attribute['DCE_PIRADS'] = roi_labels[index, att_index]
            elif roi_tag[att_index] == 'Loc':
                image_finding.attribute['Loc_PZTZ'] = roi_labels[index, att_index]
            else:
                image_finding.attribute[roi_tag[att_index]] = roi_labels[index, att_index]
        is_cancer = False
        print(image_finding.attribute.keys())
        if 'Type' in image_finding.attribute.keys():
            if image_finding.attribute['Type'] == 'CA':
                is_cancer = True
        if not is_cancer:
            roi.name = "Roi{0}".format(roi_index)
            roi_index = roi_index + 1
        else:
            roi.name = "CA{0}".format(ca_index)
            ca_index = ca_index + 1
            roi.roi_type = "CA"

        width = 0
        height = 0
        print(range(slice))
        for s in range(slice):
            slice_polygon = roi.slice_polygon.add()
            mask = label_img[index][s]
            if mask is not None:
                mask = mask.astype(np.uint8)
                print(type(mask), mask.dtype, mask.shape)
                contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours[1]) > 0:
                    count = contours[1][0].shape[0]
                    print('points', count)
                    xy_array = np.zeros(count * 2, dtype='int32')
                    xy_array[0:2 * count:2] = contours[1][0][:, 0, 0]
                    xy_array[1:2 * count:2] = contours[1][0][:, 0, 1]
                    polygon = slice_polygon.polygon.add()
                    polygon.pt = xy_array.tobytes()

            if label_img[index][s] is not None:
                # print(type(label_img[index][s]))
                width = label_img[index][s].shape[1]
                height = label_img[index][s].shape[0]
        roi.width = width
        roi.height = height
        roi.slice = slice

    str = roi_file.SerializeToString()

    with open(output_path, 'wb') as fp:
        fp.write(str)

def GetAIStationRoi(input_path):
    output_list = []

    with open(input_path, 'rb') as fp:
        str = fp.read()
        roi_file = station_roi_pb2.RoiLabelFile()
        roi_file.ParseFromString(str)

        # roi.attribute未读取，是字典的列表
        plt.ion()
        for label in roi_file.labels:
            one_label = np.zeros((label.width, label.height, label.slice))

            for slice_index, slice_polygon in enumerate(label.slice_polygon):
                # 可能在slice上的连通性导致了在某一层上不连通的两个polygon事实上是连通的
                # 但是len(slice_polygon.polygon)的典型值是0或1，大于等于2的话就是上面的提到的情况

                for polygon in slice_polygon.polygon:
                    pt_array = np.frombuffer(polygon.pt, dtype='int32')
                    a = np.zeros((label.width, label.height), dtype='uint8')
                    cv2.fillPoly(a, [pt_array.reshape((pt_array.size // 2, 2))], 1)

                    one_label[..., label.slice - slice_index - 1] = a

            output_list.append(one_label)

        return output_list, (label.width, label.height, label.slice)

####################################################################

def LoadImagejRoi(fpath):
    '''
    The following code was copyed from  https://github.com/hadim/read-roi. If I need to fork the raw source of the code
    form GitHub, please let me know.
    '''

    import struct
    import zipfile
    import logging

    OFFSET = dict(VERSION_OFFSET=4,
                  TYPE=6,
                  TOP=8,
                  LEFT=10,
                  BOTTOM=12,
                  RIGHT=14,
                  N_COORDINATES=16,
                  X1=18,
                  Y1=22,
                  X2=26,
                  Y2=30,
                  XD=18,
                  YD=22,
                  WIDTHD=26,
                  HEIGHTD=30,
                  STROKE_WIDTH=34,
                  SHAPE_ROI_SIZE=36,
                  STROKE_COLOR=40,
                  FILL_COLOR=44,
                  SUBTYPE=48,
                  OPTIONS=50,
                  ARROW_STYLE=52,
                  ELLIPSE_ASPECT_RATIO=52,
                  ARROW_HEAD_SIZE=53,
                  ROUNDED_RECT_ARC_SIZE=54,
                  POSITION=56,
                  HEADER2_OFFSET=60,
                  COORDINATES=64)

    ROI_TYPE = dict(polygon=0,
                    rect=1,
                    oval=2,
                    line=3,
                    freeline=4,
                    polyline=5,
                    noRoi=6,
                    freehand=7,
                    traced=8,
                    angle=9,
                    point=10)

    OPTIONS = dict(SPLINE_FIT=1,
                   DOUBLE_HEADED=2,
                   OUTLINE=4,
                   OVERLAY_LABELS=8,
                   OVERLAY_NAMES=16,
                   OVERLAY_BACKGROUNDS=32,
                   OVERLAY_BOLD=64,
                   SUB_PIXEL_RESOLUTION=128,
                   DRAW_OFFSET=256)

    HEADER_OFFSET = dict(C_POSITION=4,
                         Z_POSITION=8,
                         T_POSITION=12,
                         NAME_OFFSET=16,
                         NAME_LENGTH=20,
                         OVERLAY_LABEL_COLOR=24,
                         OVERLAY_FONT_SIZE=28,
                         AVAILABLE_BYTE1=30,
                         IMAGE_OPACITY=31,
                         IMAGE_SIZE=32,
                         FLOAT_STROKE_WIDTH=36)

    SUBTYPES = dict(TEXT=1,
                    ARROW=2,
                    ELLIPSE=3,
                    IMAGE=4)

    def get_byte(data, base):
        if isinstance(base, int):
            return data[base]
        elif isinstance(base, list):
            return [data[b] for b in base]

    def get_short(data, base):
        b0 = data[base]
        b1 = data[base + 1]
        n = (b0 << 8) + b1
        return n

    def get_int(data, base):
        b0 = data[base]
        b1 = data[base + 1]
        b2 = data[base + 2]
        b3 = data[base + 3]
        n = ((b0 << 24) + (b1 << 16) + (b2 << 8) + b3)
        return n

    def get_float(data, base):
        s = struct.pack('I', get_int(data, base))
        return struct.unpack('f', s)[0]


    if isinstance(fpath, zipfile.ZipExtFile):
        data = fpath.read()
        name = os.path.splitext(os.path.basename(fpath.name))[0]
    elif isinstance(fpath, str):
        fp = open(fpath, 'rb')
        data = fp.read()
        fp.close()
        name = os.path.splitext(os.path.basename(fpath))[0]
    else:
        logging.error("Can't read {}".format(fpath))
        return None

    logging.debug("Read ROI for \"{}\"".format(name))

    size = len(data)
    code = '>'

    roi = {}

    magic = get_byte(data, list(range(4)))
    magic = "".join([chr(c) for c in magic])

    # TODO: raise error if magic != 'Iout'

    version = get_short(data, OFFSET['VERSION_OFFSET'])
    roi_type = get_byte(data, OFFSET['TYPE'])
    subtype = get_short(data, OFFSET['SUBTYPE'])
    top = get_short(data, OFFSET['TOP'])
    left = get_short(data, OFFSET['LEFT'])

    if top > 6000:
        top -= 2**16
    if left > 6000:
        left -= 2**16

    bottom = get_short(data, OFFSET['BOTTOM'])
    right = get_short(data, OFFSET['RIGHT'])
    width = right - left
    height = bottom - top
    n_coordinates = get_short(data, OFFSET['N_COORDINATES'])
    options = get_short(data, OFFSET['OPTIONS'])
    position = get_int(data, OFFSET['POSITION'])
    hdr2Offset = get_int(data, OFFSET['HEADER2_OFFSET'])

    logging.debug("n_coordinates: {n_coordinates}")
    logging.debug("position: {position}")
    logging.debug("options: {options}")

    sub_pixel_resolution = (options == OPTIONS['SUB_PIXEL_RESOLUTION']) and version >= 222
    draw_offset = sub_pixel_resolution and (options == OPTIONS['DRAW_OFFSET'])
    sub_pixel_rect = version >= 223 and sub_pixel_resolution and (
        roi_type == ROI_TYPE['rect'] or roi_type == ROI_TYPE['oval'])

    logging.debug("sub_pixel_resolution: {sub_pixel_resolution}")
    logging.debug("draw_offset: {draw_offset}")
    logging.debug("sub_pixel_rect: {sub_pixel_rect}")

    # Untested
    if sub_pixel_rect:
        xd = get_float(data, OFFSET['XD'])
        yd = get_float(data, OFFSET['YD'])
        widthd = get_float(data, OFFSET['WIDTHD'])
        heightd = get_float(data, OFFSET['HEIGHTD'])
        logging.debug("Entering in sub_pixel_rect")

    # Untested
    if hdr2Offset > 0 and hdr2Offset + HEADER_OFFSET['IMAGE_SIZE'] + 4 <= size:
        channel = get_int(data, hdr2Offset + HEADER_OFFSET['C_POSITION'])
        slice = get_int(data, hdr2Offset + HEADER_OFFSET['Z_POSITION'])
        frame = get_int(data, hdr2Offset + HEADER_OFFSET['T_POSITION'])
        overlayLabelColor = get_int(data, hdr2Offset + HEADER_OFFSET['OVERLAY_LABEL_COLOR'])
        overlayFontSize = get_short(data, hdr2Offset + HEADER_OFFSET['OVERLAY_FONT_SIZE'])
        imageOpacity = get_byte(data, hdr2Offset + HEADER_OFFSET['IMAGE_OPACITY'])
        imageSize = get_int(data, hdr2Offset + HEADER_OFFSET['IMAGE_SIZE'])
        logging.debug("Entering in hdr2Offset")

    is_composite = get_int(data, OFFSET['SHAPE_ROI_SIZE']) > 0

    # Not implemented
    if is_composite:
        if version >= 218:
            pass
        if channel > 0 or slice > 0 or frame > 0:
            pass

    if roi_type == ROI_TYPE['rect']:
        roi = {'type': 'rectangle'}

        if sub_pixel_rect:
            roi.update(dict(left=xd, top=yd, width=widthd, height=heightd))
        else:
            roi.update(dict(left=left, top=top, width=width, height=height))

        roi['arc_size'] = get_short(data, OFFSET['ROUNDED_RECT_ARC_SIZE'])

    elif roi_type == ROI_TYPE['oval']:
        roi = {'type': 'oval'}

        if sub_pixel_rect:
            roi.update(dict(left=xd, top=yd, width=widthd, height=heightd))
        else:
            roi.update(dict(left=left, top=top, width=width, height=height))

    elif roi_type == ROI_TYPE['line']:
        roi = {'type': 'line'}

        x1 = get_float(data, OFFSET['X1'])
        y1 = get_float(data, OFFSET['Y1'])
        x2 = get_float(data, OFFSET['X2'])
        y2 = get_float(data, OFFSET['Y2'])

        if subtype == SUBTYPES['ARROW']:
            # Not implemented
            pass
        else:
            roi.update(dict(x1=x1, x2=x2, y1=y1, y2=y2))
            roi['draw_offset'] = draw_offset

    elif roi_type in [ROI_TYPE[t] for t in ["polygon", "freehand", "traced", "polyline", "freeline", "angle", "point"]]:
        x = []
        y = []
        base1 = OFFSET['COORDINATES']
        base2 = base1 + 2 * n_coordinates
        for i in range(n_coordinates):
            xtmp = get_short(data, base1 + i * 2)
            ytmp = get_short(data, base2 + i * 2)
            x.append(left + xtmp)
            y.append(top + ytmp)

        if sub_pixel_resolution:
            xf = []
            yf = []
            base1 = OFFSET['COORDINATES'] + 4 * n_coordinates
            base2 = base1 + 4 * n_coordinates
            for i in range(n_coordinates):
                xf.append(get_float(data, base1 + i * 4))
                yf.append(get_float(data, base2 + i * 4))

        if roi_type == ROI_TYPE['point']:
            roi = {'type': 'point'}

            if sub_pixel_resolution:
                roi.update(dict(x=xf, y=yf, n=n_coordinates))
            else:
                roi.update(dict(x=x, y=y, n=n_coordinates))

        if roi_type == ROI_TYPE['polygon']:
            roi = {'type': 'polygon'}

        elif roi_type == ROI_TYPE['freehand']:
            roi = {'type': 'freehand'}
            if subtype == SUBTYPES['ELLIPSE']:
                ex1 = get_float(data, OFFSET['X1'])
                ey1 = get_float(data, OFFSET['Y1'])
                ex2 = get_float(data, OFFSET['X2'])
                ey2 = get_float(data, OFFSET['Y2'])
                roi['aspect_ratio'] = get_float(
                    data, OFFSET['ELLIPSE_ASPECT_RATIO'])
                roi.update(dict(ex1=ex1, ey1=ey1, ex2=ex2, ey2=ey2))

        elif roi_type == ROI_TYPE['traced']:
            roi = {'type': 'traced'}

        elif roi_type == ROI_TYPE['polyline']:
            roi = {'type': 'polyline'}

        elif roi_type == ROI_TYPE['freeline']:
            roi = {'type': 'freeline'}

        elif roi_type == ROI_TYPE['angle']:
            roi = {'type': 'angle'}

        else:
            roi = {'type': 'freeroi'}

        if sub_pixel_resolution:
            roi.update(dict(x=xf, y=yf, n=n_coordinates))
            #roi.update(dict(x=x, y=y, n=n_coordinates))
        else:
            roi.update(dict(x=x, y=y, n=n_coordinates))
    else:
        # TODO: raise an error for 'Unrecognized ROI type'
        pass

    roi['name'] = name

    if version >= 218:
        # Not implemented
        # Read stroke width, stroke color and fill color
        pass

    if version >= 218 and subtype == SUBTYPES['TEXT']:
        # Not implemented
        # Read test ROI
        pass

    if version >= 218 and subtype == SUBTYPES['IMAGE']:
        # Not implemented
        # Get image ROI
        pass

    roi['position'] = position
    if channel > 0 or slice > 0 or frame > 0:
        roi['position'] = dict(channel=channel, slice=slice, frame=frame)

    return {name: roi}

def LoadImagejRoiZip(zip_path):
    import zipfile

    from collections import OrderedDict
    rois = OrderedDict()
    zf = zipfile.ZipFile(zip_path)
    for n in zf.namelist():
        rois.update(LoadImagejRoi(zf.open(n)))
    return rois

###########################################################################

def GenerateROIFromSPIN(tob_file_path, ref_image):
    '''
    This function was to read .tob file, which was designed by SPIN, MRInnovation, (by Dr. Haacke, Ying Wang.)
    :param tob_file_path: The file with '.tob'
    :param ref_image: The reference image, which provided the shape of the image
    :return: The generated ROI. Different cores was assigned to different values. The first 2 dimensions were swapped for visualization.
     By Jie Wu, Nov-11-18
    '''
    if isinstance(ref_image, str):
        ref_image = sitk.ReadImage(ref_image)

    image_shape = ref_image.GetSize()[0:3]
    recon_data = np.zeros(image_shape)

    index_list = np.fromfile(tob_file_path, dtype=np.uint32)

    total_roi = index_list[1]
    roi_number_index = 2
    for roi_index in range(total_roi):
        roi_point_number = index_list[roi_number_index] // 4
        print('The number of array: ', roi_point_number)
        for roi_point_index in range(1, roi_point_number):
            recon_data[index_list[roi_number_index + roi_point_index * 4 + 1] - 1,
                       index_list[roi_number_index + roi_point_index * 4 + 2] - 1,
                       index_list[roi_number_index + roi_point_index * 4 + 3] - 1] = roi_index + 1
        roi_number_index += roi_point_number * 4 + 1

    recon_data = np.swapaxes(recon_data, 0, 1)
    return recon_data

def ColorfulFidRead(file_path):
    with open(file_path, 'rb') as f:
        #1 double
        f.seek(0,0)
        bytes = f.read(8)
        file_version, = struct.unpack('d', bytes)
        #print(file_version)
        
        #4 int
        bytes = f.read(4*4)
        section_size = struct.unpack('4i', bytes)
        #print(section_size)    
        f.seek(8 + 4*4 + section_size[0] + section_size[1] + section_size[2], 0)
        
        #5 int
        bytes = f.read(5*4)
        dim = struct.unpack('5i', bytes)
        #print(dim)
        length = dim[4]*dim[3]*dim[2]*dim[1]*dim[0]
                
        fid = np.fromfile(f, dtype=np.complex64, count=length)
        shape = (dim[4], dim[3], dim[2], dim[1], dim[0])
        fid = fid.reshape(shape)
        
        return fid, file_version  
 
def ColorfulFidCine(data, fft=True, transpose=False):
    plt.ion()
    dim = data.shape
    data = data.reshape((-1, dim[-2],  dim[-1]))
    for step in range(data.shape[0]):  
        plt.cla()
        image = data[step, :, :]
        if fft:
            image = np.fft.fftshift(np.fft.ifftn(image))
            
        if transpose:
            plt.imshow(np.transpose(abs(image)), cmap = 'gray')
        else:
            plt.imshow(abs(image), cmap = 'gray')
        
        plt.pause(0.1)  
    plt.ioff()  
    plt.show()

if __name__ == '__main__':
    from MeDIT.OtherScript import station_roi_pb2
    data_path = r'T:\CNNFormatData\PCaDetection\T2_ADC_DWI1500_3slices\training\2012-2016-CA_formal_BSL^bai song lai ^^6698-8-9.h5'
    keys, datas = LoadH5AllTag(data_path)
    print(keys)
    print(datas)