import albumentations as A
from time import perf_counter
from torch.utils.data import Dataset
import numpy as np
import torch
import cv2
import os
import sys
import pickle 
import random


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
    def __init__(self, paths, tsize, do_aug, mean=None, std=None, min_sequence_length=1):
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
                for subdir in [s.name for s in os.scandir(os.path.join(path, label)) if s.is_dir()]:
                    frames = [os.path.join(path, label, subdir, f.name) for f in os.scandir(os.path.join(path, label, subdir))
                         if (f.is_file() and ('.jp' in f.name or '.pn' in f.name))]
                    if len(frames) >= min_sequence_length:
                        self.samples.append((i, frames[:min_sequence_length]))
                        self.targets.append(i)
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
            A.GaussNoise(p=0.5, var_limit=(1, 16)),
            A.ColorJitter(p=0.5)
        ], p=1.0, additional_targets=dict([(f'image{i}', 'image') for i in range(min_sequence_length-1)]))

    def labels_names(self):
        d = {}
        for i, label in enumerate(self.labels_list):
            d[i] = label
        return d

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tensor = []
        mats = {}
        i = 0
        for filename in sorted(self.samples[idx][1]):
            mat = cv2.imread(filename, cv2.IMREAD_COLOR)

            if mat.shape[0] != self.tsize[0] and mat.shape[1] != self.tsize[1]:
                interp = cv2.INTER_LINEAR if mat.shape[0]*mat.shape[1] > self.tsize[0]*self.tsize[1] else cv2.INTER_CUBIC
                mat = cv2.resize(mat, self.tsize, interpolation=interp)
            if i == 0:
                mats['image'] = mat
            else:
                mats[f'image{i-1}'] = mat
            i += 1

        if self.do_aug:
            mats = self.album(**mats)

        for key in mats:
            # Visual control
            #cv2.imshow("probe", mats[key])
            #cv2.waitKey(0)
            tensor.append(image2tensor(mats[key], mean=self.mean, std=self.std, swap_red_blue=True))
        tensor = np.stack(tensor)
        return torch.from_numpy(tensor), self.samples[idx][0]
    

class CustomTripletDataSet(Dataset):
    def __init__(self, paths, tsize, do_aug, mean=None, std=None, min_sequence_length=None, balanced=False):
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
        self.paths = paths
        self.seq_length = min_sequence_length
        self.balanced = balanced
        self.generate_triplets()
        self.album = A.Compose([
            A.RandomBrightnessContrast(p=0.1, brightness_limit=(-0.1, 0.1)),
            A.HorizontalFlip(p=0.5),
            A.CoarseDropout(p=0.1,
                            max_holes=3,
                            min_width=int(self.tsize[1] / 7),
                            max_width=int(self.tsize[1] / 5),
                            min_height=int(self.tsize[0] / 7),
                            max_height=int(self.tsize[0] / 5)),
            A.Affine(p=1.0,
                     scale=(0.95, 1.05),
                     translate_percent=(-0.05, 0.05),
                     rotate=(-5, 5)),
            A.GaussNoise(p=0.1, var_limit=(1, 8)),
            A.ColorJitter(p=0.1)
        ], p=1.0, additional_targets=dict([(f'image{i}', 'image') for i in range(min_sequence_length-1)]))
    
    def generate_triplets(self):

        triplets = []
        for j, path in enumerate(self.paths):
            fakes_path = os.path.join(path, 'fake')
            fakes_seqs = os.listdir(fakes_path)
            lives_path = os.path.join(path, 'live')
            lives_seqs = os.listdir(lives_path)
            print(path, len(fakes_seqs), len(lives_seqs))
            live_id2folder = dict((x.split('_')[0] if '_' in x else x.split('-@-')[0], x) for x in lives_seqs)
            if path.endswith('dfdc'):
                for i in range(2000):
                    if random.random() > 0.5:
                        anch, pos = random.sample(lives_seqs, 2)
                        neg = random.choice(fakes_seqs)
                        triplets.append((anch, pos, neg, 1, j))
                    else:
                        anch, pos = random.sample(fakes_seqs, 2)
                        neg = random.choice(lives_seqs)
                        triplets.append((anch, pos, neg, 0, j))
                continue
                
            if not self.balanced:
                #TODO fix triplets generation for roop 
                t_s = list(map(lambda x: x.split('-@-')[0], fakes_seqs))        
                triplets.extend(list(map(lambda x: tuple([live_id2folder[y] for y in x.split('_')[:2]] + [x, 1, j]), t_s)))
            else:
                for i, fake_seq in enumerate(fakes_seqs):
                    source_person_id, target_person_id = fake_seq.split('-@-')[0].split('_')[:2]
                    if random.random() > 0.5:
                        anch = live_id2folder[source_person_id] if source_person_id in live_id2folder else random.choice(lives_seqs)
                        pos = live_id2folder[target_person_id] if target_person_id in live_id2folder else random.choice(lives_seqs)
                        triplets.append((anch, pos, fake_seq, 1, j))
                    else:
                        neg_id = random.choice([source_person_id, target_person_id])
                        pos_samples_with_same_id = list(filter(lambda x: (neg_id in x) and (x!=fake_seq), fakes_seqs))
                        if len(pos_samples_with_same_id):
                            pos_sample = random.choice(pos_samples_with_same_id)
                        else:    
                            pos_sample = random.choice(fakes_seqs)
                        neg_sample = live_id2folder[neg_id] if neg_id in live_id2folder else random.choice(lives_seqs)
                        triplets.append((fake_seq, pos_sample, neg_sample, 0, j))
        
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):

        anch, pos, neg, anch_cls, path_id = self.triplets[idx]
        path = self.paths[path_id]

        anch_seq = self.load_seq(anch, anch_cls, path)
        pos_seq = self.load_seq(pos, anch_cls, path)
        neg_seq = self.load_seq(neg, 1 - anch_cls, path)
        
        return anch_seq, pos_seq, neg_seq, anch_cls

    def load_seq(self, folder, label, path):
        
        folder_path = os.path.join(path, 'live' if label else 'fake', folder)
        filenames = os.listdir(folder_path)
        tensor = []
        mats = dict()
        if len(filenames) > self.seq_length:
            start_idx = random.randint(0, len(filenames) - self.seq_length)
            end_idx = start_idx + self.seq_length
            filenames = filenames[start_idx: end_idx]
        elif len(filenames) < self.seq_length:
            while len(filenames) < self.seq_length:
                filenames += filenames
            filenames = filenames[:self.seq_length]

        for i, filename in enumerate(filenames):
            file_path = os.path.join(folder_path, filename)
            mat = cv2.imread(file_path, cv2.IMREAD_COLOR)

            if mat.shape[0] != self.tsize[0] and mat.shape[1] != self.tsize[1]:
                interp = cv2.INTER_LINEAR if mat.shape[0]*mat.shape[1] > self.tsize[0]*self.tsize[1] else cv2.INTER_CUBIC
                mat = cv2.resize(mat, self.tsize, interpolation=interp)

            if i == 0:
                mats['image'] = mat
            else:
                mats[f'image{i-1}'] = mat
        
        if self.do_aug:
            mats = self.album(**mats)

        for key in mats:
            tensor.append(image2tensor(mats[key], mean=self.mean, std=self.std, swap_red_blue=True))
        tensor = np.stack(tensor)
        
        return torch.from_numpy(tensor)
    
    def get_image(self, file_path):

        mat = cv2.imread(file_path, cv2.IMREAD_COLOR)
        mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)

        if mat.shape[0] != self.tsize[0] and mat.shape[1] != self.tsize[1]:
            interp = cv2.INTER_LINEAR if mat.shape[0]*mat.shape[1] > self.tsize[0]*self.tsize[1] else cv2.INTER_CUBIC
            mat = cv2.resize(mat, self.tsize, interpolation=interp)

        if self.do_aug:
            mat = self.album(image=mat)["image"]

        return torch.from_numpy(image2tensor(mat, mean=self.mean, std=self.std))



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



if __name__ == '__main__':
    dataset = CustomTripletDataSet(['../sequence/256x60x0.1/toloka'], 
                                   tsize=(256, 256), 
                                   do_aug=True, 
                                   min_sequence_length=10, 
                                   balanced=True)
    print(dataset.triplets[:2])
    anch, pos, neg, anch_cls = dataset.__getitem__(0)
    print(anch.shape, pos.shape, neg.shape, anch_cls)