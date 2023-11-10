from imgutils import fit_img_into_rectangle, BoundingBox
import cv2


class FaceDetector:
    # Base abstraction for all kinds of possible face detectors
    # Should return list of FaceBoundingBox instances
    def detect(self, img):
        raise NotImplemented


class YuNetFaceDetector(FaceDetector):  # https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet
    def __init__(self, filename, input_size=72, threshold=0.9):
        self.input_size = (input_size, input_size)
        self.model = cv2.FaceDetectorYN_create(model=filename,
                                               config='',
                                               input_size=self.input_size,
                                               score_threshold=threshold)

    def detect(self, img):
        resized, scale, shifts = fit_img_into_rectangle(img,
                                                        target_width=self.input_size[0],
                                                        target_height=self.input_size[0])
        _, pred = self.model.detect(resized)
        bboxes = []
        if pred is not None:
            for item in pred:
                raw = item[:4]
                bbox = BoundingBox((raw[0] - shifts[1]) / scale,
                                   (raw[1] - shifts[0]) / scale,
                                   raw[2] / scale,
                                   raw[3] / scale)
                bboxes.append(bbox)
        return bboxes
