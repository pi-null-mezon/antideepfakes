import torch
import torch.jit
from imgutils import extract_face_fixed_eyes_distance, image2tensor
from landmarks import LWADetector
from face import FaceDetector
import numpy as np
import cv2
import os


class VideoFileSource:
    def __init__(self, cap, filename, delete_file=True):
        self.filename = filename
        self.strobe = 1
        self.counter = 0
        self.cap = cap
        self.delete_file = delete_file
        assert self.cap.isOpened()

    def frames_total(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def frames_last(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - self.counter

    def set_strobe(self, strobe):
        self.strobe = strobe

    def next(self):
        for i in range(self.strobe):
            ret, frame = self.cap.read()
            self.counter += 1
            if not ret:
                if self.delete_file:
                    os.remove(self.filename)
                raise StopIteration
        return f"frame_{(self.counter-1):06d}", frame


class FaceVideoProcessor:
    # Base abstraction for all kinds of possible face processors
    def process(self, video: VideoFileSource, fdet: FaceDetector, ldet: LWADetector, strobe: int, max_sequences: int):
        sequences = self.prepare_input_batch(video, fdet, ldet, strobe, max_sequences)
        if sequences is None:  # no face found
            return None
        scores = []
        with torch.no_grad():
            for model in self.sequence_models:
                prediction = model(sequences)
                scores.append(prediction.mean(dim=0)[1].item())
        return np.array(scores).mean().item()


class AlignedCropsProcessor(FaceVideoProcessor):
    def __init__(self, width, height, eyes_dst, rotate, v2hshift, mean, std, interp, swap_red_blue, sequence_length,
                 device):
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
        self.sequence_length = sequence_length
        self.device = device

    def prepare_input_batch(self, video: VideoFileSource, fdet: FaceDetector, ldet: LWADetector, strobe: int, max_sequences: int):
        crops = []
        video.set_strobe(strobe)
        while True:
            try:
                name, frame = video.next()
            except StopIteration:
                break
            if isinstance(frame, np.ndarray):
                bboxes = fdet.detect(frame)
                if len(bboxes) > 0:
                    bboxes.sort(key=lambda x: x.area(), reverse=True)
                    angles, landmarks = ldet.process(frame, [bboxes[0]])
                    crop, pts = extract_face_fixed_eyes_distance(frame, landmarks[0], self.eyes_dst,
                                                                 (self.width, self.height), self.rotate, self.v2hshift)
                    crops.append(crop)

        if len(crops) == 0:
            return None

        tensors = []
        for crop in crops:
            # DEBUG VISUALIZATION
            # cv2.imshow('crop', crop)
            # cv2.waitKey(1)
            tensors.append(image2tensor(crop, self.mean, self.std, self.swap_red_blue))
        tensors = torch.from_numpy(np.stack(tensors))  # frames x channels x heights x width

        if tensors.shape[0] < self.sequence_length:
            while tensors.shape[0] < self.sequence_length:
                tensors = torch.cat([tensors, tensors[(tensors.shape[0] - self.sequence_length):]])
            sequences = torch.empty(size=(1, self.sequence_length, 3, self.height, self.width))
            sequences[0] = tensors
        else:
            step = tensors.shape[0] // self.sequence_length
            torch.manual_seed(12112023)
            sequences = torch.empty(size=(min(max_sequences, step), self.sequence_length, 3, self.height, self.width))
            for i in range(sequences.shape[0]):
                sequence = tensors[i + torch.randint(0, step, (1,))::step][:self.sequence_length]
                if sequence.shape[0] == self.sequence_length:
                    sequences[i] = sequence
                else:
                    sequences[i] = torch.cat([sequence, sequence[(sequence.shape[0] - self.sequence_length):]])
        if self.device == torch.device('cuda'):
            sequences = sequences.half()
        return sequences.to(self.device)


class DD256x60x01(AlignedCropsProcessor):
    def __init__(self, filenames: list, device):
        AlignedCropsProcessor.__init__(self, width=256, height=256, eyes_dst=60.0, rotate=True, v2hshift=-0.1,
                                       mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                                       interp=cv2.INTER_LINEAR, swap_red_blue=True,
                                       sequence_length=10,
                                       device=device)
        self.sequence_models = []
        for filename in filenames:
            model = torch.jit.load(filename).to(self.device)
            if self.device == torch.device('cuda'):
                model.half()
            model.eval()
            self.sequence_models.append(model)


class DD224x90x02(AlignedCropsProcessor):
    def __init__(self, filenames: list, device):
        AlignedCropsProcessor.__init__(self, width=224, height=224, eyes_dst=90.0, rotate=True, v2hshift=-0.2,
                                       mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                                       interp=cv2.INTER_LINEAR, swap_red_blue=True,
                                       sequence_length=10,
                                       device=device)
        self.sequence_models = []
        for filename in filenames:
            model = torch.jit.load(filename).to(self.device)
            if self.device == torch.device('cuda'):
                model.half()
            model.eval()
            self.sequence_models.append(model)


# Returns liveness score if video file was decoded successfully and face has been found (even on single frame)
# In case of file decode error raises IOError
# In case of no face has been found returns None
def liveness_score(filename: str, dds: list, fd: FaceDetector, ld: LWADetector, strobe: int, max_sequences: int,
                   delete_file: bool):
    scores = []
    for index in range(len(dds)):
        cap = cv2.VideoCapture(filename)
        if not cap.isOpened():
            raise IOError
        video = VideoFileSource(cap, filename, delete_file=True if (delete_file and index == (len(dds) - 1)) else False)
        score = dds[index].process(video, fd, ld, strobe, max_sequences)
        if score is None:
            return None
        scores.append(score)
    return np.array(scores).mean().item()
