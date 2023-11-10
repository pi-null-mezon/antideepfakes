from imgutils import extract_face_fixed_eyes_distance, normalize_image
from landmarks import LWADetector
from face import FaceDetector
from numpy import np
import cv2


class VideoFileSource:
    def __init__(self, filename, strobe):
        self.filename = filename
        self.strobe = strobe
        self.counter = 0
        self.cap = cv2.VideoCapture(filename)
        assert self.cap.isOpened(), f"Can not open '{filename}'!"

    def next(self):
        for i in range(self.strobe):
            ret, frame = self.cap.read()
            self.counter += 1
            if not ret:
                raise StopIteration
        return f"frame_{self.counter:06d}", frame


class FaceVideoProcessor:
    # Base abstraction for all kinds of possible face processors
    def process(self, video: VideoFileSource, fdet: FaceDetector, ldet: LWADetector):
        raise NotImplemented


class AlignedCropsProcessor(FaceVideoProcessor):
    def __init__(self, width, height, eyes_dst, rotate, v2hshift, mean, std, interp, swap_red_blue):
        FaceVideoProcessor.__init__(self)
        self.width = width
        self.height = height
        self.eyes_dst = eyes_dst
        self.rotate = rotate
        self.v2hshift = v2hshift
        self.mean = mean
        self.std = std
        self.interp = interp
        self.swap_red_blue = swap_red_blue

    def prepare_input_batch(self, video: VideoFileSource, fdet: FaceDetector, ldet: LWADetector):
        cap = cv2.VideoCapture("video.mp4")
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(length)

        tensors = []
        for img, lmks in zip(imgs, landmarks):
            crop, pts = extract_face_fixed_eyes_distance(img, lmks, self.eyes_dst, (self.width, self.height),
                                                         self.rotate, self.v2hshift)
            # DEBUG VISUALIZATION
            # for pt in pts:
            #    cv2.circle(crop, (int(pt.x), int(pt.y)), 1, (0, 255, 0), -1, cv2.LINE_AA)
            # cv2.imshow('crop', crop)
            # cv2.waitKey(0)
            tensors.append(normalize_image(crop, self.mean, self.std, self.swap_red_blue))
        return cv2.dnn.blobFromImages(images=tensors)


class DeepfakeDetector(AlignedCropsProcessor):
    def __init__(self, filename):
        AlignedCropsProcessor.__init__(self,
                                       width=96,
                                       height=112,
                                       eyes_dst=37.0,
                                       rotate=True,
                                       v2hshift=-0.025,
                                       mean=3 * [0.5],
                                       std=3 * [0.501960784],
                                       interp=cv2.INTER_LINEAR,
                                       swap_red_blue=False)
        self.model = cv2.dnn.readNetFromONNX(filename)


    def process(self, video: VideoFileSource, fdet: FaceDetector, ldet: LWADetector):
        input_blob = self.prepare_input_batch(video, fdet, ldet)
        self.model.setInput(input_blob)
        output_blob = self.model.forward()
        return np.squeeze(output_blob, axis=1).tolist()
