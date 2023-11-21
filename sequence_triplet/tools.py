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


class ROCEstimator:
    def __init__(self):
        self.live_scores = []
        self.attack_scores = []
        self._apcer_curve = None
        self._bpcer_curve = None

    def reset(self):
        self.live_scores = []
        self.attack_scores = []
        self._apcer_curve = None
        self._bpcer_curve = None

    def update(self, live_scores: list, attack_scores: list):
        self.live_scores += live_scores
        self.attack_scores += attack_scores

    def apcer_curve(self, steps_total=1E3, epsilon=1.0E-6, force_recalc=True):
        if force_recalc:
            self._apcer_curve = None
        if self._apcer_curve is not None:
            return self._apcer_curve
        self._apcer_curve = []
        numpy_scores = np.asarray(self.attack_scores)
        total_outcomes = len(numpy_scores)
        for threshold in np.linspace(start=0.0 - epsilon, stop=1.0 + epsilon, num=int(steps_total)):
            self._apcer_curve.append(np.sum(numpy_scores > threshold) / total_outcomes)
        return self._apcer_curve

    def bpcer_curve(self, steps_total=1E3, epsilon=1.0E-6, force_recalc=True):
        if force_recalc:
            self._bpcer_curve = None
        if self._bpcer_curve is not None:
            return self._bpcer_curve
        self._bpcer_curve = []
        numpy_scores = np.asarray(self.live_scores)
        total_outcomes = len(numpy_scores)
        for threshold in np.linspace(start=0.0 - epsilon, stop=1.0 + epsilon, num=int(steps_total)):
            self._bpcer_curve.append(np.sum(numpy_scores <= threshold) / total_outcomes)
        return self._bpcer_curve

    def estimate_eer(self, steps_total=1E3, epsilon=1.0E-6):
        attack_pts = self.apcer_curve(steps_total, epsilon, force_recalc=False)
        live_pts = self.bpcer_curve(steps_total, epsilon, force_recalc=False)
        index = 0
        for i in range(0, len(attack_pts)):
            index = i
            if (attack_pts[i] - live_pts[i]) <= 0:
                break
        return (attack_pts[index] + live_pts[index]) / 2.0, index / steps_total

    def estimate_bpcer(self, target_apcer, steps_total=1E3, epsilon=1.0E-6):
        attack_pts = self.apcer_curve(steps_total, epsilon, force_recalc=False)
        live_pts = self.bpcer_curve(steps_total, epsilon, force_recalc=False)
        for i in range(0, len(attack_pts) - 1):
            if (attack_pts[i] - target_apcer) * (attack_pts[i + 1] - target_apcer) <= 0:
                return live_pts[i + 1] - (target_apcer - attack_pts[i + 1]) * \
                       (live_pts[i + 1] - live_pts[i]) / (attack_pts[i] - attack_pts[i + 1])
        return live_pts[len(live_pts) - 1]


def read_img_as_torch_tensor(filename, size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], swap_rb=True):
    mat = cv2.imread(filename, cv2.IMREAD_COLOR)
    assert (mat.shape[0] == size[0] and mat.shape[1] == size[1]), "sizes missmatch"
    return torch.from_numpy(image2tensor(mat, mean=mean, std=std, swap_red_blue=swap_rb))


def check_absolute_difference(t1: torch.Tensor, t2: torch.Tensor, eps: float = 1.0E-6):
    diff = torch.abs(t1 - t2).sum().item()
    return diff < eps


def benchmark_inference(model, device, input_tensor_shape, warmup_iters, work_iters, about=''):
    print(f"benchmarking model: {about}")
    model.to(torch.device(device))
    print(f" - device: {device}")
    tmp = torch.randn(size=input_tensor_shape).to(device)
    print(f" - input shape: {input_tensor_shape}")
    print(f" - warmup iterations: {warmup_iters}")
    print(f" - work iterations: {work_iters}")
    with torch.no_grad():
        for i in range(warmup_iters):
            model(tmp)
        accum = []
        for i in range(work_iters):
            t0 = perf_counter()
            model(tmp)
            accum.append(perf_counter() - t0)
        accum = torch.Tensor(accum)
    print(f" - duration (95 %): {1000.0*accum.mean().item():.2f} ± {2*1000.0*accum.std().item():.2f} ms")