import albumentations as A
from time import perf_counter
from torch.utils.data import Dataset
import numpy as np
import torch
import cv2
import os
import sys


def normalize_image(bgr, mean, std, swap_red_blue=False):
    tmp = bgr.astype(dtype=np.float32) / 255.0
    if swap_red_blue:
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
    tmp -= np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    tmp /= np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    return tmp


def image2tensor(bgr, mean, std, swap_red_blue=False):
    tmp = normalize_image(bgr, mean, std, swap_red_blue)
    return np.transpose(tmp, axes=(2, 0, 1))  # HxWxC -> CxHxW


class CustomDataSet(Dataset):
    def __init__(self, paths, tsize, do_aug, mean=None, std=None):
        """
            :param tsize: target size of images
            :param do_aug: enable augmentations
            :param mean: values to substract
            :param std: values to normalize on
        """
        if mean is None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std is None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std
        self.tsize = tsize
        self.do_aug = do_aug
        self.samples = []
        self.targets = []
        for path in paths:
            self.labels_list = [s.name for s in os.scandir(path) if s.is_dir()]
            self.labels_list.sort()
            for i, label in enumerate(self.labels_list):
                files = [(i, os.path.join(path, label, f.name)) for f in os.scandir(os.path.join(path, label))
                         if (f.is_file() and ('.jp' in f.name or '.pn' in f.name))]
                self.samples += files
                self.targets += [i]*len(files)
        self.album = A.Compose([
            A.RandomBrightnessContrast(p=0.25, brightness_limit=(-0.25, 0.25)),
            A.HorizontalFlip(p=0.5),
            A.CoarseDropout(p=0.75,
                            max_holes=3,
                            min_width=int(self.tsize[1] / 7),
                            max_width=int(self.tsize[1] / 5),
                            min_height=int(self.tsize[0] / 7),
                            max_height=int(self.tsize[0] / 5)),
            A.Affine(p=1.0,
                     scale=(0.95, 1.05),
                     translate_percent=(-0.05, 0.05),
                     rotate=(-5, 5)),
            #A.GaussNoise(p=0.5, var_limit=(1, 35)),
            #A.ImageCompression(p=0.25, quality_lower=70, quality_upper=100),
            #A.RandomBrightnessContrast(p=0.25, brightness_limit=(-0.25, 0.25)),
            A.ColorJitter(p=1.0)
        ], p=1.0)

    def labels_names(self):
        d = {}
        for i, label in enumerate(self.labels_list):
            d[i] = label
        return d

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename = self.samples[idx][1]
        mat = cv2.imread(filename, cv2.IMREAD_COLOR)
        mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)

        if mat.shape[0] != self.tsize[0] and mat.shape[1] != self.tsize[1]:
            interp = cv2.INTER_LINEAR if mat.shape[0]*mat.shape[1] > self.tsize[0]*self.tsize[1] else cv2.INTER_CUBIC
            mat = cv2.resize(mat, self.tsize, interpolation=interp)

        if self.do_aug:
            mat = self.album(image=mat)["image"]

        # Visual control
        # cv2.imshow("probe", mat)
        # cv2.waitKey(0)
        return torch.from_numpy(image2tensor(mat, mean=self.mean, std=self.std)), self.samples[idx][0]


class Speedometer:
    def __init__(self, gamma: float = 0.9):
        self.gamma = gamma
        self._speed = None
        self.t0 = perf_counter()

    def reset(self):
        self._speed = None
        self.t0 = perf_counter()

    def update(self, samples):
        if self._speed is None:
            self._speed = samples / (perf_counter() - self.t0)
        else:
            self._speed = self._speed * self.gamma + (1 - self.gamma) * samples / (perf_counter() - self.t0)
        self.t0 = perf_counter()

    def speed(self):
        return self._speed


class Averagemeter:
    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
        self.val = None

    def reset(self):
        self.val = None

    def update(self, val):
        if self.val is None:
            self.val = val
        else:
            self.val = self.val * self.gamma + (1 - self.gamma) * val

    def value(self):
        return self.val


def print_one_line(s):
    sys.stdout.write('\r' + s)
    sys.stdout.flush()


def model_size_mb(model):
    params_size = 0
    for param in model.parameters():
        params_size += param.nelement() * param.element_size()
    buffers_size = 0
    for buffer in model.buffers():
        buffers_size += buffer.nelement() * buffer.element_size()
    return (params_size + buffers_size) / 1024 ** 2
