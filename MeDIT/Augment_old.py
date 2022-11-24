"""
All rights reserved.
-- Yang Song.
"""
from abc import abstractmethod
import random
from copy import deepcopy

import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

from MeDIT.ArrayProcess import Crop2DArray

################################################################
def multi_variable(func):
    def wrapper(*args, **kwargs):
        assert (len(args) == 2)
        if not isinstance(args[1], list):
            inputs = (args[0], [args[1]])
        else:
            inputs = args
        class_self = inputs[0]
        class_inputs = inputs[1]

        if 'skip' not in kwargs.keys() or kwargs['skip'] is None:
            kwargs['skip'] = [False for index in class_inputs]

        results = []
        class_self.RandomParam()
        for one_input, one_skip in zip(class_inputs, kwargs['skip']):
            if one_skip and class_self.skip_roi:
                results.append(one_input)
            else:
                results.append(func(class_self, one_input))

        if not isinstance(args[1], list):
            results = results[0]
        return results

    return wrapper


class BaseTransform(object):
    def __init__(self, random_config=None, skip_roi=False):
        self.random_config = random_config
        self.skip_roi = skip_roi

    @abstractmethod
    def __call__(self, data):
        raise NotImplementedError

    def __str__(self):
        return str(type(self).__name__)

    def ParamGenerator(self, config, random_state=random, cast_type=None):
        if config[0] == 'uniform':
            ret = random_state.uniform(config[1], config[2])
        elif config[0] == 'lognormal':
            ret = random_state.lognormvariate(config[1], config[2])
        elif config[0] == 'choice':
            ret = random_state.choice(config[1:])
        elif config[0] == 'elastix_matrix':
            pass
        else:
            print(config)
            raise Exception('unsupported format')

        if cast_type is None:
            return ret
        else:
            return cast_type(ret)

    @abstractmethod
    def RandomParam(self):
        raise NotImplementedError


class RotateTransform(BaseTransform):
    def __init__(self, theta=0., intep_method=cv2.INTER_LINEAR, **kwargs):
        super().__init__(skip_roi=False, **kwargs)
        self.theta = theta
        self.interp_method = intep_method

    def RandomParam(self):
        if self.random_config is not None:
            assert('theta' in self.random_config.keys())
            self.theta = self.ParamGenerator(self.random_config['theta'])

    @multi_variable
    def __call__(self, data):
        assert(data.ndim == 2)
        row, col = data.shape
        alpha = np.cos(self.theta / 180 * np.pi)
        beta = np.sin(self.theta / 180 * np.pi)

        matrix = np.array([
            [alpha, beta, (1 - alpha) * (col // 2) - beta * (row // 2)],
            [-beta, alpha, beta * (col // 2) + (1 - alpha) * (row // 2)]
        ])
        return cv2.warpAffine(data, matrix, (col, row), flags=self.interp_method + cv2.WARP_FILL_OUTLIERS)


class ShiftTransform(BaseTransform):
    #TODO: 考虑不同尺度大小进行成比例的shift操作
    def __init__(self, shear=0., horizontal_shift=0., vertical_shift=0.,
                 interp_method=cv2.INTER_LINEAR, **kwargs):
        super().__init__(skip_roi=False, **kwargs)
        self.shear = shear
        self.horizontal_shift = horizontal_shift
        self.vertical_shift = vertical_shift
        self.interp_method = interp_method

    def RandomParam(self):
        if self.random_config is not None:
            if 'shear' in self.random_config.keys():
                self.shear = self.ParamGenerator(self.random_config['shear'])
            if 'horizontal_shift' in self.random_config.keys():
                self.horizontal_shift = self.ParamGenerator(self.random_config['horizontal_shift'])
            if 'vertical_shift' in self.random_config.keys():
                self.vertical_shift = self.ParamGenerator(self.random_config['vertical_shift'])

    @multi_variable
    def __call__(self, data):
        assert(data.ndim == 2)
        row, col = data.shape
        matrix = np.array([
            [1., 0., self.horizontal_shift * col],
            [self.shear, 1., self.vertical_shift * row]
        ])

        result = cv2.warpAffine(data, matrix, (col, row), flags=self.interp_method + cv2.WARP_FILL_OUTLIERS)
        return result


class ZoomTransform(BaseTransform):
    def __init__(self, horizontal_zoom=1., vertical_zoom=1., interp_method=cv2.INTER_LINEAR, **kwargs):
        super().__init__(skip_roi=False, **kwargs)
        self.horizontal_zoom = horizontal_zoom
        self.vertical_zoom = vertical_zoom
        self.interp_method = interp_method

    def RandomParam(self):
        if self.random_config is not None:
            if 'horizontal_zoom' in self.random_config.keys():
                self.horizontal_zoom = self.ParamGenerator(self.random_config['horizontal_zoom'])
            if 'vertical_zoom' in self.random_config.keys():
                self.vertical_zoom = self.ParamGenerator(self.random_config['vertical_zoom'])

    @multi_variable
    def __call__(self, data):
        assert(data.ndim == 2)
        result = cv2.resize(data, None, fx=self.vertical_zoom, fy=self.horizontal_zoom,
                            interpolation=self.interp_method)
        result = Crop2DArray(result, data.shape)
        return result


class FlipTransform(BaseTransform):
    def __init__(self, horizontal_flip=False, vertical_flip=False, **kwargs):
        super().__init__(skip_roi=False, **kwargs)
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

    def RandomParam(self):
        if self.random_config is not None:
            if 'horizontal_flip' in self.random_config.keys():
                self.horizontal_flip = self.ParamGenerator(self.random_config['horizontal_flip'])
            if 'vertical_flip' in self.random_config.keys():
                self.vertical_flip = self.ParamGenerator(self.random_config['vertical_flip'])

    @multi_variable
    def __call__(self, data):
        assert(data.ndim == 2)
        result = deepcopy(data)
        if self.horizontal_flip:
            result = np.flip(result, axis=1)
        if self.vertical_flip:
            result = np.flip(result, axis=0)
        return result


class BiasTransform(BaseTransform):
    # center范围-1到1
    def __init__(self, center=(0., 0.), drop_ratio=0., **kwargs):
        super().__init__(skip_roi=True, **kwargs)
        # 将范围转到0~1
        self.center = (np.clip((center[0] + 1) / 2, a_min=0., a_max=1.),
                       np.clip((center[1] + 1) / 2, a_min=0., a_max=1.))
        self.drop_ratio = drop_ratio

    def _BiasField(self, input_shape):
        # field = a * (x - center_x) ** 2 + b * (y - center_y) ** 2 + 1
        field = np.zeros(shape=input_shape)

        center_x = int(input_shape[0] * self.center[0])
        center_y = int(input_shape[1] * self.center[1])
        max_x = max(input_shape[0] - center_x, center_x)
        max_y = max(input_shape[1] - center_y, center_y)

        a = -self.drop_ratio / (2 * max_x ** 2)
        b = -(self.drop_ratio + a * max_x ** 2) / (max_y ** 2)

        row, column = np.meshgrid(range(field.shape[1]), range(field.shape[0]))
        field = a * (row - center_x) ** 2 + b * (column - center_y) ** 2 + 1
        return field

    def RandomParam(self):
        if self.random_config is not None:
            if 'center' in self.random_config.keys():
                self.center = (self.ParamGenerator(self.random_config['center']),
                               self.ParamGenerator(self.random_config['center']))
            if 'drop_ratio' in self.random_config.keys():
                self.drop_ratio = self.ParamGenerator(self.random_config['drop_ratio'])

    @multi_variable
    def __call__(self, data):
        assert(data.ndim == 2)
        field = self._BiasField(data.shape)
        min_value = data.min()
        result = np.multiply(field, data - min_value) + min_value
        return result


class ElasticTransform(BaseTransform):
    def __init__(self, raw_shape=(256, 256), alpha=None, sigma=None, **kwargs):
        super().__init__(skip_roi=False, **kwargs)
        # alpha = 3, sigma = 0.1
        self.raw_shape = raw_shape
        self._GenerateBias(alpha, sigma)

    def _GenerateBias(self, alpha, sigma):
        if (alpha is not None and sigma is not None):
            self.dx = gaussian_filter((np.random.rand(*self.raw_shape) * 2 - 1),
                                 sigma * min(self.raw_shape), mode="constant", cval=0) * alpha * min(self.raw_shape)
            self.dy = gaussian_filter((np.random.rand(*self.raw_shape) * 2 - 1),
                                 sigma * min(self.raw_shape), mode="constant", cval=0) * alpha * min(self.raw_shape)
        else:
            self.dx = self.dy = None

    def RandomParam(self):
        if self.random_config is not None:
            if 'alpha' in self.random_config.keys():
                alpha = self.ParamGenerator(self.random_config['alpha'])
            else:
                alpha = None
            if 'sigma' in self.random_config.keys():
                sigma = self.ParamGenerator(self.random_config['sigma'])
            else:
                sigma = None

            self._GenerateBias(alpha, sigma)

    @multi_variable
    def __call__(self, data):
        assert(data.ndim == 2)
        if self.dx is None or self.dy is None:
            result = data
        else:
            if data.shape != self.raw_shape:
                temp = cv2.resize(data, self.raw_shape, interpolation=cv2.INTER_LINEAR)
            else:
                temp = data
            assert(self.raw_shape == temp.shape)
            x, y = np.meshgrid(np.arange(self.raw_shape[0]), np.arange(self.raw_shape[1]), indexing='ij')
            indices = [np.reshape(x + self.dx, (-1, 1)), np.reshape(y + self.dy, (-1, 1))]
            result = map_coordinates(temp, indices, order=1, mode='nearest').reshape(self.raw_shape)

            if data.shape != self.raw_shape:
                result = cv2.resize(result, data.shape, interpolation=cv2.INTER_NEAREST)
        return result


class ElasticTransform2D(BaseTransform):
    def __init__(self, dx, dy, **kwargs):
        super().__init__(skip_roi=False, **kwargs)
        self.dx = dx
        self.dy = dy
        assert(dx.shape == dy.shape)

    def GenerateBias(self, alpha, sigma, shape=(512, 512)):
        self.dx = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                                  sigma * min(shape), mode="constant", cval=0) * alpha * min(shape)
        self.dy = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                                  sigma * min(shape), mode="constant", cval=0) * alpha * min(shape)

    def RandomParam(self):
        if self.random_config is not None:
            if 'alpha' in self.random_config.keys():
                alpha = self.ParamGenerator(self.random_config['alpha'])
            else:
                alpha = None
            if 'sigma' in self.random_config.keys():
                sigma = self.ParamGenerator(self.random_config['sigma'])
            else:
                sigma = None

            self.GenerateBias(alpha, sigma)

    @multi_variable
    def __call__(self, data):
        assert(data.ndim == 2)
        if self.dx is None or self.dy is None:
            result = data
        else:
            shape = self.dx.shape
            if data.shape != self.dx.shape:
                temp = cv2.resize(data, shape, interpolation=cv2.INTER_LINEAR)
            else:
                temp = data
            assert(shape == temp.shape)
            x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
            indices = [np.reshape(x + self.dx, (-1, 1)), np.reshape(y + self.dy, (-1, 1))]
            result = map_coordinates(temp, indices, order=1, mode='nearest').reshape(shape)

            if data.shape != shape:
                result = cv2.resize(result, data.shape, interpolation=cv2.INTER_NEAREST)
        return result


# 不要求相同参数，得出图像不同
class NoiseTransform(BaseTransform):
    def __init__(self, noise_sigma=0., **kwargs):
        super().__init__(skip_roi=True, **kwargs)
        self.noise_sigma = noise_sigma

    def RandomParam(self):
        if self.random_config is not None:
            if 'noise_sigma' in self.random_config.keys():
                self.noise_sigma = self.ParamGenerator(self.random_config['noise_sigma'])

    @multi_variable
    def __call__(self, data):
        assert(data.ndim == 2)
        noise = np.random.normal(0., self.noise_sigma * data.max(), data.shape)
        result = (data + noise).astype(data.dtype)
        return result


class ContrastTransform(BaseTransform):
    def __init__(self, factor=1., preserve_range=True, **kwargs):
        super().__init__(skip_roi=True, **kwargs)
        self.factor = factor
        self.preserve_range = preserve_range

    def RandomParam(self):
        if self.random_config is not None:
            if 'factor' in self.random_config.keys():
                self.factor = self.ParamGenerator(self.random_config['factor'])

    @multi_variable
    def __call__(self, data):
        assert(data.ndim == 2)
        mean_value = data.mean()
        result = (data - mean_value) * self.factor + mean_value
        if self.preserve_range:
            result = np.clip(result, a_min=data.min(), a_max=data.max())
        return result


class GammaTransform(BaseTransform):
    def __init__(self, gamma=1.0, **kwargs):
        super().__init__(skip_roi=True, **kwargs)
        self.gamma = gamma

    def RandomParam(self):
        if self.random_config is not None:
            if 'gamma' in self.random_config.keys():
                self.gamma = self.ParamGenerator(self.random_config['gamma'])

    @multi_variable
    def __call__(self, data):
        assert(data.max() - data.min() > 1e-6)
        result = np.power((data - data.min()) / (data.max() - data.min()), self.gamma) *\
                 (data.max() - data.min()) + data.min()
        return result


def TestRotate(data, skip=None):
    if isinstance(data, np.ndarray):
        result = []
        for theta in np.arange(0, 360, 10):
            trans = RotateTransform(theta=theta)
            result.append(trans(data))
        FlattenImages(result, is_show=True)

        result = []
        trans = RotateTransform(random_config={'theta': ['uniform', -45, 45]})
        for index in range(16):
            result.append(trans(data))
        FlattenImages(result, is_show=True)
    else:
        results = [[] for index in data]
        for theta in np.arange(0, 360, 10):
            trans = RotateTransform(theta=theta)
            ones = trans(data, skip=skip)
            for index, one in enumerate(ones):
                results[index].append(one)
        for result in results:
            FlattenImages(result, is_show=True)

        results = [[] for index in data]
        trans = RotateTransform(random_config={'theta': ['uniform', -45, 45]})
        for index in range(16):
            ones = trans(data, skip=skip)
            for index, one in enumerate(ones):
                results[index].append(one)
        for result in results:
            FlattenImages(result, is_show=True)

def TestShift(data):
    result = []
    for x in np.arange(-0.5, 0.5, 0.5):
        for y in np.arange(-0.2, 0.2, 0.2):
            trans = ShiftTransform(horizontal_shift=x, vertical_shift=y)
            result.append(trans(data))
    FlattenImages(result, is_show=True)

def TestZoom(data):
    result = []
    result.append(data)
    for x in np.arange(0.5, 1.5, 0.3):
        for y in np.arange(0.5, 1.5, 0.3):
            trans = ZoomTransform(zoom=(x, y))
            result.append(trans(data))
    FlattenImages(result, is_show=True)

def TestBias(data):
    result = []
    for center_x in np.arange(-1., 1.1, 0.5):
        for center_y in np.arange(-1., 1.1, 0.5):
            for drop in [0.]:
                trans = BiasTransform((center_x, center_y), drop)
                result.append(trans(data))
    FlattenImages(result, is_show=True)

def TestFlip(data):
    result = []
    for flip in [(True, True), (True, False), (False, True), (False, False)]:
        trans = FlipTransform(horizontal=flip[0], vertical=flip[1])
        result.append(trans(data))
    FlattenImages(result, is_show=True)

def TestElastic(data, skip=None):
    if isinstance(data, np.ndarray):
        result = []
        trans = ElasticTransform(alpha=5, sigma=0.1)
        for index in range(16):
            result.append(trans(data))
        FlattenImages(result, is_show=True)
    else:
        results = [[] for index in data]
        trans = ElasticTransform(alpha=5, sigma=0.1)
        for index in range(16):
            ones = trans(data, skip=skip)
            for index, one in enumerate(ones):
                results[index].append(one)
        for result in results:
            FlattenImages(result, is_show=True)

    results = [[] for index in data]
    trans = ElasticTransform(random_config={'alpha': ['uniform', 3, 3],
                                            'sigma': ['uniform', 0.1, 0.1]})
    for index in range(16):
        ones = trans(data, skip=skip)
        for index, one in enumerate(ones):
            results[index].append(one)
    for result in results:
        FlattenImages(result, is_show=True)

def TestNoise(data):
    result = []
    trans = NoiseTransform(sigma=0.01)
    for index in range(16):
        result.append(trans(data))
    FlattenImages(result, is_show=True)

def TestContrast(data):
    result = []
    for factor in np.arange(0.7, 1.3, 0.1):
        trans = ContrastTransform(factor)
        result.append(trans(data))
    FlattenImages(result, is_show=True)

def TestGamma(data):
    result = []
    for factor in np.arange(0.7, 1.3, 0.1):
        trans = GammaTransform(factor)
        result.append(trans(data))
    FlattenImages(result, is_show=True)


class TransformManager(object):
    def __init__(self, config=None):
        self.transform_sequence = config

    def AddOneTransform(self, transform):
        self.transform_sequence.append(transform)

    def Transform(self, data, skip=False):
        if not isinstance(data, list):
            data = [data]
        if not isinstance(skip, list):
            skip = [skip]

        if self.transform_sequence is None:
            return data
        else:
            temp = deepcopy(data)
            for transform in self.transform_sequence:
                temp = transform(temp, skip=skip)

            if len(data) == 1:
                temp = temp[0]
            return temp


random_config = [
    RotateTransform(random_config={'theta': ['uniform', -20, 10]}),
    ShiftTransform(random_config={'horizontal_shift': ['uniform', -0.1, 0.1],
                                  'vertical_shift': ['uniform', -0.1, 0.1]}),
    ZoomTransform(random_config={'horizontal_zoom': ['uniform', 0.9, 1.1],
                                 'vertical_zoom': ['uniform', 0.9, 1.1]}),
    FlipTransform(random_config={'horizontal_flip': ['choice', True, False]}),
    BiasTransform(random_config={'center': ['uniform', -1., 1.],
                                 'drop_ratio': ['uniform', 0., 1.]}),
    NoiseTransform(random_config={'noise_sigma': ['uniform', 0., 0.03]}),
    ContrastTransform(random_config={'factor': ['uniform', 0.8, 1.2]}),
    GammaTransform(random_config={'gamma': ['uniform', 0.8, 1.2]}),
    ElasticTransform(random_config={'alpha': ['uniform', 2.8, 3.2],
                                    'sigma': ['uniform', 0.1, 0.1]})
]


def TestTransform(data, skip):
    trans = TransformManager(random_config)
    # trans = TransformManager()
    results = [[], []]
    for index in range(25):
        result = trans.Transform(data, skip)
        results[0].append(result[0])
        results[1].append(result[1])

    for result in results:
        FlattenImages(result, True)

if __name__ == '__main__':
    from MeDIT.UsualUse import *
    from MeDIT.Visualization import FlattenImages

    print(RotateTransform().__str__())

    _, d, _ = LoadImage(r'd:\Data\HouYing\processed\BIAN JIN YOU\t2.nii')
    data1 = d[:, 100:-100, d.shape[2] // 2]
    data2 = d[100:-100, :, d.shape[2] // 2]
    print(data1.shape)
    # TestShift(data1)
    TestTransform([d[..., d.shape[2] // 2], d[..., d.shape[2] // 2]], [False, True])


    # print(data2.shape)
    # TestElastic([data1, data2], skip=[False, False])
