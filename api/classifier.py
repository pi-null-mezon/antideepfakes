import torch
import numpy as np
import cv2


class FaceClassifierAbstract:
    """
        Base class for all types of possible face processors / classifiers
    """

    def __init__(self, weights: str, device: str,  mean: list, std: list, interp: int, swap_red_blue: bool):
        self.mean = mean
        self.std = std
        self.interp = interp
        self.swap_red_blue = swap_red_blue
        self.device = device
        if '.pth' in weights:
            self.backend = 'torch'
            self.device = torch.device('cuda' if (torch.cuda.is_available() and self.device == 'cuda') else 'cpu')
            self.model = torch.load(weights).module.to(device)
            self.model.eval()
        #elif '.onnx' in weights:
        #    self.backend = 'onnx'
        #    if self.device == 'cuda':
        #        self.model = onnxruntime.InferenceSession(weights, providers=['CUDAExecutionProvider'])
        #    else:
        #        self.model = onnxruntime.InferenceSession(weights, providers=['CPUExecutionProvider'])
        elif '.cv2' in weights:
            self.backend = 'opencv'
            self.model = cv2.dnn.readNetFromONNX(weights)
            if self.device == 'cuda':
                # you should provide cv2 builded with OPENCV_DNN_CUDA=ON
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("FaceClassifier instance created:")
        print(f" - weights: '{weights}'")
        print(f" - backend: {self.backend}")
        print(f" - device: {self.device}")

    def prepare_input_batch(self, img, boxes_or_landmarks):
        raise NotImplementedError

    def process(self, img, bboxes):
        if self.backend == 'torch':
            with torch.no_grad():
                # t0 = time.perf_counter()
                batch = self.prepare_input_batch(img, bboxes).to(self.device)
                # print(f" batch preparation: {(time.perf_counter() - t0)*1000.0:.2f} ms")
                # t0 = time.perf_counter()
                prediction = self.model(batch)
                # print(f" inference: {(time.perf_counter() - t0) * 1000.0:.2f} ms")
                return prediction
        elif self.backend == 'onnx':
            # t0 = time.perf_counter()
            batch = self.prepare_input_batch(img, bboxes)
            # print(f" batch preparation: {(time.perf_counter() - t0) * 1000.0:.2f} ms")
            # t0 = time.perf_counter()
            out = self.model.run(None, {self.model.get_inputs()[0].name: batch})[0]
            # print(f" inference: {(time.perf_counter() - t0) * 1000.0:.2f} ms")
            return out
        elif self.backend == 'opencv':
            # t0 = time.perf_counter()
            input_blob = self.prepare_input_batch(img, bboxes)
            self.model.setInput(input_blob)
            # print(f" batch preparation: {(time.perf_counter() - t0) * 1000.0:.2f} ms")
            # t0 = time.perf_counter()
            output_blob = self.model.forward()
            # print(f" inference: {(time.perf_counter() - t0) * 1000.0:.2f} ms")
            return output_blob
        else:
            raise NotImplementedError


class FaceBboxClassifier(FaceClassifierAbstract):
    """
        Base class for all types of possible face processors / classifiers
        Any descendants should provide implementation of at least one of:
         - process(img, List[bboxes]) - method that should return list of predicted classes/features of the faces
    """

    def __init__(self, weights: str, device: str, size: int, bbox_upscale: float, mean: list, std: list, interp: int,
                 swap_red_blue: bool):
        FaceClassifierAbstract.__init__(self, weights, device, mean, std, interp, swap_red_blue)
        self.labels = None
        self.size = size
        self.bbox_upscale = bbox_upscale

    def prepare_input_batch(self, img, bboxes):
        if self.backend == 'opencv':
            images = []
            for bbox in bboxes:
                crop, sbox = extract_face_square(img, bbox, self.bbox_upscale, self.size, self.interp)
                images.append(normalize_image(crop, self.mean, self.std, self.swap_red_blue))
            return cv2.dnn.blobFromImages(images=images)
        else:
            tensors = []
            for bbox in bboxes:
                crop, sbox = extract_face_square(img, bbox, self.bbox_upscale, self.size, self.interp)
                tensors.append(image2tensor(crop, self.mean, self.std, self.swap_red_blue))
            tensors = np.stack(tensors)
            if self.backend == 'torch':
                return torch.from_numpy(tensors)
            return tensors
        
        
class AlignedFaceClassifier(FaceClassifierAbstract):
    """
        Extract face by landmarks
    """

    def __init__(self, weights: str, device: str, width: int, height: int, eyes_dst: float, rotate: bool,
                 v2hshift: float, mean: list, std: list, interp: int, swap_red_blue: bool):
        FaceClassifierAbstract.__init__(self, weights, device, mean, std, interp, swap_red_blue)
        self.size = (width, height)
        self.eyes_dst = eyes_dst
        self.rotate = rotate
        self.v2hshift = v2hshift
        
    def prepare_input_batch(self, img, landmarks):
        if self.backend == 'opencv':
            images = []
            for lmks in landmarks:
                crop, _ = extract_aligned_face(img, lmks, self.eyes_dst, self.size, self.rotate, self.v2hshift)
                images.append(normalize_image(crop, self.mean, self.std, self.swap_red_blue))
            return cv2.dnn.blobFromImages(images=images)
        else:
            tensors = []
            for lmks in landmarks:
                crop, _ = extract_aligned_face(img, lmks, self.eyes_dst, self.size, self.rotate, self.v2hshift)
                cv2.imshow('crop', crop)
                tensors.append(image2tensor(crop, self.mean, self.std, self.swap_red_blue))
            tensors = np.stack(tensors)
            if self.backend == 'torch':
                return torch.from_numpy(tensors)
            return tensors


class BoundingBox:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def area(self):
        return self.w * self.h

    def __and__(self, other):
        bbox = BoundingBox(0, 0, 0, 0)
        if other.x <= self.x < (other.x + other.w):
            bbox.x = self.x
            bbox.w = min(other.x + other.w - self.x, self.w)
        elif self.x <= other.x < (self.x + self.w):
            bbox.x = other.x
            bbox.w = min(self.x + self.w - other.x, other.w)
        if other.y <= self.y < (other.y + other.h):
            bbox.y = self.y
            bbox.h = min(other.y + other.h - self.y, self.h)
        elif self.y <= other.y < (self.y + self.h):
            bbox.y = other.y
            bbox.h = min(self.y + self.h - other.y, other.h)
        return bbox
    
    
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        return Point(self.x * other, self.y * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __floordiv__(self, other):
        return Point(self.x / other, self.y / other)

    def __truediv__(self, other):
        return Point(self.x / other, self.y / other)
    
    
class FaceLandmarks:
    def __init__(self, pts):
        self.pts = pts

    def size(self):
        return len(self.pts)

    def five(self, ovision_convention=True):
        points = []
        if len(self.pts) == 68:
            tmp = np.array([[self.pts[36].x, self.pts[36].y],
                            [self.pts[37].x, self.pts[37].y],
                            [self.pts[38].x, self.pts[38].y],
                            [self.pts[39].x, self.pts[39].y],
                            [self.pts[40].x, self.pts[40].y],
                            [self.pts[41].x, self.pts[41].y]]).mean(axis=0)
            points.append(Point(tmp[0], tmp[1]))
            tmp = np.array([[self.pts[42].x, self.pts[42].y],
                            [self.pts[43].x, self.pts[43].y],
                            [self.pts[44].x, self.pts[44].y],
                            [self.pts[45].x, self.pts[45].y],
                            [self.pts[46].x, self.pts[46].y],
                            [self.pts[47].x, self.pts[47].y]]).mean(axis=0)
            points.append(Point(tmp[0], tmp[1]))
            points.append(self.pts[30])
            points.append(self.pts[48])
            points.append(self.pts[54])
        else:
            raise ValueError(f"Can not convert {len(self.pts)} to five points!")
        if ovision_convention:
            return np.array([[points[0].x, points[0].y],
                             [points[1].x, points[1].y],
                             [points[2].x, points[2].y],
                             [points[3].x, points[3].y],
                             [points[4].x, points[4].y]]).round().astype(np.int32)
        return points


def square_bbox(bbox: BoundingBox, upscale: float) -> BoundingBox:
    if bbox.w > bbox.h:
        rect = BoundingBox(bbox.x + (bbox.w - bbox.h) / 2.0, bbox.y, bbox.h, bbox.h)
    else:
        rect = BoundingBox(bbox.x, bbox.y + (bbox.h - bbox.w) / 2.0, bbox.w, bbox.w)
    return BoundingBox(rect.x - rect.w * (upscale - 1.0) / 2.0,
                       rect.y - rect.h * (upscale - 1.0) / 2.0,
                       rect.w * upscale, rect.h * upscale)


def extract_face_square(img, bbox: BoundingBox, upscale: float, target_size: int, interpolation: int):
    frame_bbox = BoundingBox(0, 0, img.shape[1], img.shape[0])
    sbox = square_bbox(bbox, upscale) & frame_bbox
    return crop_img_from_center_and_resize(img[int(sbox.y):int(sbox.y + sbox.h), int(sbox.x):int(sbox.x + sbox.w)],
                                           (target_size, target_size), interpolation=interpolation)


def crop_img_from_center_and_resize(img, target_size, interpolation):
    bbox = BoundingBox(0, 0, 0, 0)
    target_rows = target_size[0]
    target_cols = target_size[1]
    if img.shape[0] / img.shape[1] > target_rows / target_cols:
        bbox.w = img.shape[1]
        bbox.h = bbox.w * target_rows / target_cols
        bbox.y = (img.shape[0] - bbox.h) / 2.0
    else:
        bbox.h = img.shape[0]
        bbox.w = bbox.h * target_cols / target_rows
        bbox.x = (img.shape[1] - bbox.w) / 2.0
    bbox = bbox & BoundingBox(0, 0, img.shape[1], img.shape[0])
    tmp = img[int(bbox.y):int(bbox.y + bbox.h), int(bbox.x):int(bbox.x + bbox.w)]
    out = cv2.resize(tmp, target_size, 0, 0, interpolation=interpolation)
    return out, bbox


def extract_aligned_face(img, landmarks, target_eyes_dst, target_size, rotate, v2hshift, h2wshift=0,
                                     interpolation: int = cv2.INTER_LINEAR):
    five = landmarks.five(ovision_convention=False)
    lpt = five[0]
    rpt = five[1]
    eyes_dst = np.sqrt(np.square(lpt.x - rpt.x) + np.square(lpt.y - rpt.y))
    nose = (lpt + rpt) / 2 - five[2]
    nose_length = np.sqrt(np.square(nose.x) + np.square(nose.y)) + 1E-6
    angle = 180.0 * np.arctan((lpt.y - rpt.y) / (lpt.x - rpt.x)) / np.pi if rotate else 0.0
    cpt = (rpt + lpt) / 2
    scale = target_eyes_dst / (eyes_dst + 1.0E-7)
    if eyes_dst / nose_length < 1.0:
        scale = target_eyes_dst / (1.25 * nose_length)
        angle = 0
    rm = cv2.getRotationMatrix2D((cpt.x, cpt.y), angle, scale)
    rm[0, 2] += target_size[0] / 2.0 - cpt.x + h2wshift * target_size[0]
    rm[1, 2] += target_size[1] / 2.0 - cpt.y + v2hshift * target_size[1]
    aligned_img = cv2.warpAffine(img, rm, target_size, flags=interpolation)
    aligned_pts = []
    if isinstance(landmarks, np.ndarray):
        for pt in [Point(landmarks[i, 0], landmarks[i, 1]) for i in range(landmarks.shape[0])]:
            aligned_pts.append(Point(pt.x * rm[0, 0] + pt.y * rm[0, 1] + rm[0, 2],
                                     pt.x * rm[1, 0] + pt.y * rm[1, 1] + rm[1, 2]))
    elif isinstance(landmarks, FaceLandmarks):
        for pt in landmarks.pts:
            aligned_pts.append(Point(pt.x * rm[0, 0] + pt.y * rm[0, 1] + rm[0, 2],
                                     pt.x * rm[1, 0] + pt.y * rm[1, 1] + rm[1, 2]))
    return aligned_img, aligned_pts


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


def numpy_softmax(x, axis):
    ex = np.exp(x)
    sm = np.expand_dims(np.sum(ex, axis=axis), axis=1)
    return ex / (sm + 1.E-6)
