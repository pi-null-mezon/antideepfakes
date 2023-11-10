import numpy as np
import cv2


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

    def __floordiv__(self, other):
        return Point(self.x / other, self.y / other)

    def __truediv__(self, other):
        return Point(self.x / other, self.y / other)


class FaceLandmarks:
    def __init__(self, pts):
        self.pts = pts

    def size(self):
        return len(self.pts)

    def five(self):
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
        return points


class HeadAngles:  # in degrees
    def __init__(self, pitch, yaw, roll):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll


def fit_img_into_rectangle(img, target_width, target_height, interpolation=cv2.INTER_LINEAR):
    output = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    if (target_width / target_height) > (img.shape[1] / img.shape[0]):
        height = target_height
        width = int(target_height * (img.shape[1] / img.shape[0]))
    else:
        width = target_width
        height = int(target_width * (img.shape[0] / img.shape[1]))
    scale = width / img.shape[1]
    shift_cols = (target_width - width) // 2
    shift_rows = (target_height - height) // 2
    output[shift_rows:(shift_rows + height), shift_cols:(shift_cols + width)] = \
        cv2.resize(img, (width, height), interpolation=interpolation)
    return output, scale, (shift_rows, shift_cols)


def square_bbox(bbox: BoundingBox, upscale: float) -> BoundingBox:
    if bbox.w > bbox.h:
        rect = BoundingBox(bbox.x + (bbox.w - bbox.h) / 2.0, bbox.y, bbox.h, bbox.h)
    else:
        rect = BoundingBox(bbox.x, bbox.y + (bbox.h - bbox.w) / 2.0, bbox.w, bbox.w)
    return BoundingBox(rect.x - rect.w * (upscale - 1.0) / 2.0,
                       rect.y - rect.h * (upscale - 1.0) / 2.0,
                       rect.w * upscale, rect.h * upscale)


def crop_img_from_center_and_resize(img, target_size, interpolation=cv2.INTER_LINEAR):
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


def crop_landmarks(landmarks, frame_bbox, bbox, upscale, roi, target_side_size):
    lmks = []
    square = square_bbox(bbox, upscale) & frame_bbox
    resize_x = target_side_size / roi.w
    resize_y = target_side_size / roi.h
    for pt in landmarks.pts:
        lmks.append(Point((pt.x - square.x - roi.x) * resize_x, (pt.y - square.y - roi.y) * resize_y))
    return FaceLandmarks(lmks)


def fit_landmarks(landmarks, scale, shifts):
    lmks = []
    for pt in landmarks.pts:
        lmks.append(Point((pt.x * scale) + shifts[1], (pt.y * scale) + shifts[0]))
    return FaceLandmarks(lmks)


def extract_face_square(img, bbox: BoundingBox, upscale: float, target_size: int,
                        interpolation: int = cv2.INTER_LINEAR):
    frame_bbox = BoundingBox(0, 0, img.shape[1], img.shape[0])
    sbox = square_bbox(bbox, upscale) & frame_bbox
    return crop_img_from_center_and_resize(img[int(sbox.y):int(sbox.y + sbox.h),
                                           int(sbox.x):int(sbox.x + sbox.w)],
                                           (target_size, target_size),
                                           interpolation=interpolation)


def extract_face_fixed_eyes_distance(img, landmarks, target_eyes_dst, target_size, rotate, v2hshift=0, h2wshift=0,
                                     interpolation: int = cv2.INTER_LINEAR):
    five = landmarks.five()
    lpt = five[0]
    rpt = five[1]
    eyes_dst = np.sqrt(np.square(lpt.x - rpt.x) + np.square(lpt.y - rpt.y))
    nose = (lpt + rpt) / 2 - five[2]
    nose_length = np.sqrt(np.square(nose.x) + np.square(nose.y)) + 1E-6
    angle = 180.0 * np.arctan((lpt.y - rpt.y) / (lpt.x - rpt.x)) / np.pi if rotate else 0.0
    cpt = (rpt + lpt) / 2
    scale = target_eyes_dst / eyes_dst
    if eyes_dst / nose_length < 1.0:
        scale = target_eyes_dst / (1.25 * nose_length)
        angle = 0
    rmatrix = cv2.getRotationMatrix2D((cpt.x, cpt.y), angle, scale)
    rmatrix[0, 2] += target_size[0] / 2.0 - cpt.x + h2wshift * target_size[0]
    rmatrix[1, 2] += target_size[1] / 2.0 - cpt.y + v2hshift * target_size[1]
    aligned_img = cv2.warpAffine(img, rmatrix, target_size, flags=interpolation)
    points = []
    for pt in landmarks.pts:
        points.append(Point(pt.x * rmatrix[0, 0] + pt.y * rmatrix[0, 1] + rmatrix[0, 2],
                            pt.x * rmatrix[1, 0] + pt.y * rmatrix[1, 1] + rmatrix[1, 2]))
    return aligned_img, FaceLandmarks(points)


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