"""
All rights reserved.
-- Yang Song.
"""
from abc import abstractmethod, ABCMeta
import collections, random
from copy import deepcopy

import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

from MeDIT.ArrayProcess import Crop2DArray


################################################################
class BaseTransform(object):
    def __init__(self, random_config=None, skip_roi=False, raw_param=None, func=None):
        self.random_config = random_config
        self.skip_roi = skip_roi
        self.func = func
        self.raw_param = raw_param
        self.name = func.__name__

    def __call__(self, data, param=None, skip=None):
        # 设置变换参数
        used_param = {}
        if param is None:
            param = {}
        for key, value in self.raw_param.items():
            if key in param.keys():
                used_param[key] = param[key]
            else:
                used_param[key] = self.raw_param[key]

        # 判断data是否多个
        if not isinstance(data, list):
            used_data = [deepcopy(data)]
        else:
            used_data = deepcopy(data)

        # 设置是否跳过（例如ROI）
        if skip is None:
            used_skip = [False for index in used_data]
        elif isinstance(skip, bool):
            used_skip = [skip]
        else:
            used_skip = skip

        assert(len(used_skip) == len(used_data))

        result = []
        for one_data, one_skip in zip(used_data, used_skip):
            if one_skip and self.skip_roi:
                result.append(one_data)
            else:
                result.append(self.func(one_data, used_param))

        if len(result) == 1:
            result = result[0]

        return result


def Rotate(data, param):
    row, col = data.shape
    alpha = np.cos(param['theta'] / 180 * np.pi)
    beta = np.sin(param['theta'] / 180 * np.pi)

    matrix = np.array([
        [alpha, beta, (1 - alpha) * (col // 2) - beta * (row // 2)],
        [-beta, alpha, beta * (col // 2) + (1 - alpha) * (row // 2)]
    ])
    return cv2.warpAffine(data, matrix, (col, row), flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
RotateTransform = BaseTransform(func=Rotate, raw_param={'theta': 0})


def Shift(data, param):
    row, col = data.shape
    matrix = np.array([
        [1., 0., param['horizontal_shift'] * col],
        [param['shear'], 1., param['horizontal_shift'] * row]
    ])
    result = cv2.warpAffine(data, matrix, (col, row), flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    return result
ShiftTransform = BaseTransform(func=Shift, raw_param={'horizontal_shift': 0.,
                                                      'vertical_shift': 0.,
                                                      'shear': 0.})

def Zoom(data, param):
    result = cv2.resize(data, None, fx=param['vertical_zoom'], fy=param['horizontal_zoom'],
                        interpolation=cv2.INTER_LINEAR)
    result = Crop2DArray(result, data.shape)
    return result
ZoomTransform = BaseTransform(func=Zoom, raw_param={'vertical_zoom': 1., 'horizontal_zoom': 1.})


def Flip(data, param):
    result = deepcopy(data)
    if param['horizontal_flip']:
        result = np.flip(result, axis=1)
    if param['vertical_flip']:
        result = np.flip(result, axis=0)
    return result
FlipTransform = BaseTransform(func=Flip, raw_param={'horizontal_flip': False,
                                                    'vertical_flip': False})

def Bias(data, param):
    center = (np.clip((param['center'][0] + 1) / 2, a_min=0., a_max=1.),
              np.clip((param['center'][1] + 1) / 2, a_min=0., a_max=1.))

    center_x = int(data.shape[0] * center[0])
    center_y = int(data.shape[1] * center[1])
    max_x = max(data.shape[0] - center_x, center_x)
    max_y = max(data.shape[1] - center_y, center_y)

    a = -param['drop_ratio'] / (2 * max_x ** 2)
    b = -(param['drop_ratio'] + a * max_x ** 2) / (max_y ** 2)

    row, column = np.meshgrid(range(data.shape[1]), range(data.shape[0]))
    field = a * (row - center_x) ** 2 + b * (column - center_y) ** 2 + 1
    min_value = data.min()
    result = np.multiply(field, data - min_value) + min_value
    return result
BiasTransform = BaseTransform(func=Bias, skip_roi=True, raw_param={'drop_ratio': 0.,
                                                                   'center': (0., 0.)})

def Elastic(data, param):
    if param['dx'] is None or param['dy'] is None:
        result = data
    else:
        if data.shape != param['shape']:
            temp = cv2.resize(data, param['shape'], interpolation=cv2.INTER_LINEAR)
        else:
            temp = data
        assert (param['shape'] == temp.shape)
        x, y = np.meshgrid(np.arange(param['shape'][0]), np.arange(param['shape'][1]), indexing='ij')
        indices = [np.reshape(x + param['dx'], (-1, 1)), np.reshape(y + param['dy'], (-1, 1))]
        result = map_coordinates(temp, indices, order=1, mode='nearest').reshape(param['shape'])

        if data.shape != param['shape']:
            result = cv2.resize(result, data.shape, interpolation=cv2.INTER_NEAREST)
    return result
ElasticTransform = BaseTransform(func=Elastic, raw_param={'dx': None, 'dy': None,
                                                          'shape': (256, 256)})


def Noise(data, param):
    if data.max() - data.min() < 1e-6:
        return data

    if data.max() < 0.:
        new_data = data - data.min()
        noise = np.random.normal(0., param['noise_sigma'] * new_data.max(), data.shape)
    else:
        noise = np.random.normal(0., param['noise_sigma'] * data.max(), data.shape)
    result = (data + noise).astype(data.dtype)
    return result
# 不要求相同参数，得出图像不同
NoiseTransform = BaseTransform(func=Noise, skip_roi=True, raw_param={'noise_sigma': 0.})


def Contrast(data, param):
    if data.max() - data.min() < 1e-6:
        return data
    mean_value = data.mean()
    result = (data - mean_value) * param['factor'] + mean_value
    result = np.clip(result, a_min=data.min(), a_max=data.max())
    return result
ContrastTransform = BaseTransform(func=Contrast, skip_roi=True, raw_param={'factor': 1.})


def Gamma(data, param):
    if data.max() - data.min() < 1e-6:
        return data
    result = np.power((data - data.min()) / (data.max() - data.min()), param['gamma']) * \
             (data.max() - data.min()) + data.min()
    return result
GammaTransform = BaseTransform(func=Gamma, skip_roi=True, raw_param={'gamma': 1.})


class ParamGenerate(object):
    def __init__(self, config):
        self.config = config
        self.__method_list = [
            RotateTransform,
            ShiftTransform,
            FlipTransform,
            ZoomTransform,
            BiasTransform,
            ElasticTransform,
            NoiseTransform,
            ContrastTransform,
            GammaTransform
        ]

    def Random(self, config_value, random_state=random, cast_type=None):
        if config_value[0] == 'uniform':
            if len(config_value[1:]) == 2:
                ret = random_state.uniform(config_value[1], config_value[2])
            elif len(config_value[1:]) == 3:
                return tuple([random_state.uniform(config_value[1], config_value[2]) for _ in range(config_value[3])])
            else:
                print('Uniform should have 2 or 3 elements.')
                raise AssertionError
        elif config_value[0] == 'lognormal':
            ret = random_state.lognormvariate(config_value[1], config_value[2])
        elif config_value[0] == 'choice':
            ret = random_state.choice(config_value[1:])
        elif config_value[0] == 'elastic':
            alpha, sigma, shape = config_value[1:4]
            if isinstance(shape, int):
                shape = (shape, shape)
            dx = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                                 sigma * min(shape), mode="constant", cval=0) * alpha * min(shape)
            dy = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                                 sigma * min(shape), mode="constant", cval=0) * alpha * min(shape)
            return dx, dy
        else:
            print(config_value)
            raise Exception('unsupported format')

        if cast_type is None:
            return ret
        else:
            return cast_type(ret)

    def Generate(self):
        param = {}
        for method in self.__method_list:
            if method.name in self.config.keys():
                param[method.name] = {}
                if method.name == 'Elastic':
                    param[method.name]['dx'], param[method.name]['dy'] = \
                        self.Random(self.config[method.name])
                else:
                    for key, value in self.config[method.name].items():
                        param[method.name][key] = self.Random(value)
        return param


class TransformManager(object):
    def __init__(self, transform_sequence=None, data_shape=None):
        self.transform_sequence = []
        if transform_sequence is not None:
            for one_transform in transform_sequence:
                self.transform_sequence.append(one_transform)

    def Transform(self, data, param=None, skip=None):
        temp = deepcopy(data).astype(float)
        for transform in self.transform_sequence:
            if transform.name in param.keys():
                temp = transform(temp, param=param[transform.name], skip=skip)
            else:
                temp = transform(temp, skip=skip)
        return temp


def TestParamGenerate():
    pg = ParamGenerate(config=config_example)
    for index in range(10):
        print(pg.Generate())

def TestTransform(data, skip):
    pg = ParamGenerate({
        RotateTransform.name: {'theta': ['uniform', -20, 20]},
        FlipTransform.name: {'horizontal_flip': ['choice', True, False]},
        ContrastTransform.name: {'factor': ['uniform', 0.8, 1.2]},
        GammaTransform.name: {'gamma': ['uniform', 0.8, 1.2]},
        ElasticTransform.name: ['elastic', 4, 0.1, 256]
                        })

    trans = TransformManager(transform_sequence=[RotateTransform,
                                                 FlipTransform,
                                                 ContrastTransform,
                                                 GammaTransform,
                                                 ElasticTransform])
    # trans = TransformManager(transform_sequence=[ElasticTransform])

    results = [[], []]
    for _ in tqdm(range(9)):
        param = pg.Generate()
        print(param[ElasticTransform.name]['dx'][:10, :10])
        results[0].append(trans.Transform(data[0], skip=skip[0], param=param))
        results[1].append(trans.Transform(data[1], skip=skip[1], param=param))


    for result in results:
        FlattenImages(result, True)

config_example = {
    RotateTransform.name: {'theta': ['uniform', -20, 20]},
    ShiftTransform.name: {'horizontal_shift': ['uniform', -0.1, 0.1],
                          'vertical_shift': ['uniform', -0.1, 0.1]},
    ZoomTransform.name: {'horizontal_zoom': ['uniform', 0.9, 1.1],
                         'vertical_zoom': ['uniform', 0.9, 1.1]},
    FlipTransform.name: {'horizontal_flip': ['choice', True, False]},
    BiasTransform.name: {'center': ['uniform', -1., 1., 2],
                         'drop_ratio': ['uniform', 0., 1.]},
    NoiseTransform.name: {'noise_sigma': ['uniform', 0., 0.03]},
    ContrastTransform.name: {'factor': ['uniform', 0.8, 1.2]},
    GammaTransform.name: {'gamma': ['uniform', 0.8, 1.2]},
    ElasticTransform.name: ['elastic', 1, 0.1, 256]
}

map_dict = {
    RotateTransform.name: RotateTransform,
    ShiftTransform.name: ShiftTransform,
    ZoomTransform.name: ZoomTransform,
    FlipTransform.name: FlipTransform,
    BiasTransform.name: BiasTransform,
    NoiseTransform.name: NoiseTransform,
    ContrastTransform.name: ContrastTransform,
    GammaTransform.name: GammaTransform,
    ElasticTransform.name: ElasticTransform
}

get_sequence = lambda param: [map_dict[key] for key in param.keys() if key in map_dict.keys()]

if __name__ == '__main__':
    from tqdm import tqdm
    from MeDIT.Visualization import FlattenImages
    from MeDIT.SaveAndLoad import LoadImage
    _, d, _ = LoadImage(r'd:\Data\HouYing\processed\BIAN JIN YOU\t2.nii')
    data1 = d[:, 100:-100, d.shape[2] // 2]
    data2 = d[100:-100, :, d.shape[2] // 2]
    print(data1.shape, data2.shape)

    TestTransform([d[..., d.shape[2] // 2], d[..., d.shape[2] // 2]], [False, True])

    # TestParamGenerate()
