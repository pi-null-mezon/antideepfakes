from imgutils import BoundingBox, square_bbox, crop_img_from_center_and_resize, HeadAngles, Point, FaceLandmarks
import numpy as np
import cv2


class FaceProcessor:
    def process(self, img, coordinates):
        raise NotImplemented


class LWADetector(FaceProcessor):

    def __init__(self, filename):
        self.bbox_upscale = 1.9
        self.input_size = (100, 100)
        self.model = cv2.dnn.readNetFromONNX(filename)
        self.output_layers_names = [self.model.getLayerNames()[n - 1] for n in self.model.getUnconnectedOutLayers()]

    def process(self, img, bboxes):
        angles = []
        landmarks = []
        if len(bboxes) > 0:
            frame_bbox = BoundingBox(0, 0, img.shape[1], img.shape[0])
            images = []
            enlarged_bboxes = []
            rois = []
            for i in range(len(bboxes)):
                bbox = square_bbox(bboxes[i], upscale=self.bbox_upscale) & frame_bbox
                fimg, roi = crop_img_from_center_and_resize(img[int(bbox.y):int(bbox.y + bbox.h),
                                                            int(bbox.x):int(bbox.x + bbox.w)],
                                                            self.input_size)
                images.append(fimg.astype(np.float32))
                enlarged_bboxes.append(bbox)
                rois.append(roi)
            # trained with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            input_blob = cv2.dnn.blobFromImages(images=images,
                                                scalefactor=1 / 57.12,
                                                size=(0, 0),  # save input spatial dimensions
                                                mean=np.array([123.675, 116.28, 103.53]),
                                                swapRB=True,  # BGR -> RGB
                                                crop=False)
            self.model.setInput(input_blob)
            first_output_blob, second_output_blob = self.model.forward(self.output_layers_names)
            if self.output_layers_names[0] == 'angles':
                raw_angles = first_output_blob
                raw_landmarks = second_output_blob
            else:
                raw_angles = second_output_blob
                raw_landmarks = first_output_blob
            angles = []
            landmarks = []
            for i, bbox in enumerate(enlarged_bboxes):
                angles.append(HeadAngles(pitch=-90.0 * raw_angles[i][0],
                                         yaw=-90.0 * raw_angles[i][1],
                                         roll=-90.0 * raw_angles[i][2]))
                points = []
                for j in range(len(raw_landmarks[i]) // 2):
                    points.append(Point((0.5 + raw_landmarks[i][2 * j]) * rois[i].w + rois[i].x + bbox.x,
                                        (0.5 + raw_landmarks[i][2 * j + 1]) * rois[i].h + rois[i].y + bbox.y))
                landmarks.append(FaceLandmarks(points))
        return angles, landmarks
