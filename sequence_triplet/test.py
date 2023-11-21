import os
import sys
import torch
from tqdm import tqdm
from collections import Counter
from easydict import EasyDict as edict
from dataset import CustomDataSet
from tools import ROCEstimator
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------- CONFIG -----------

os.environ['TORCH_HOME'] = './weights'

cfg = edict()
cfg.crop_size = (256, 256)
cfg.sequence_length = 10  # samples to be selected per sequence
cfg.batch_size = 8 * 8#4       # sequences to be selected in minibatch
crop_format = '256x60x0.1' if cfg.crop_size[0] == 256 else '224x90x0.2'
local_path = f'../sequence/{crop_format}'


dataloaders = dict()
print("train dataset:")
train_dataset = CustomDataSet([f'{local_path}/FaceForensics++',
                             f'{local_path}/roop'
                            ],
    tsize=cfg.crop_size,
    do_aug=False,
    min_sequence_length=cfg.sequence_length)
print(f"  {train_dataset.labels_names()}")
print(f"  {dict(Counter(train_dataset.targets))}")
dataloaders['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                              drop_last=False, num_workers=16)

print("val dataset:")
val_dataset = CustomDataSet([
    f"{local_path}/Celeb-DF-v2"
],
    tsize=cfg.crop_size,
    do_aug=False,
    min_sequence_length=cfg.sequence_length)
print(f"  {val_dataset.labels_names()}")
print(f"  {dict(Counter(val_dataset.targets))}")
dataloaders['Celeb-DF-v2'] = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True,
                                              drop_last=False, num_workers=16)
print("Test dataset:")
test_dataset = CustomDataSet([
    f"{local_path}/toloka"
],
    tsize=cfg.crop_size,
    do_aug=False,
    min_sequence_length=cfg.sequence_length)
print(f"  {test_dataset.labels_names()}")
print(f"  {dict(Counter(test_dataset.targets))}")
dataloaders['toloka'] = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True,
                                              drop_last=False, num_workers=16)

print("Test1 dataset:")
test1_dataset = CustomDataSet([
    f"{local_path}/toloka_enh"
],
    tsize=cfg.crop_size,
    do_aug=False,
    min_sequence_length=cfg.sequence_length)
print(f"  {test_dataset.labels_names()}")
print(f"  {dict(Counter(test1_dataset.targets))}")
dataloaders['toloka_enh'] = torch.utils.data.DataLoader(test1_dataset, batch_size=cfg.batch_size, shuffle=True,
                                              drop_last=False, num_workers=16)




def test(epoch, dataloader, mode, backbone, model, alive_lbl=1):
    print(f"{mode.upper()}:")

    roc_est = ROCEstimator()
    model.eval()
    backbone.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, file=sys.stdout)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.view(-1, 3, inputs.shape[-1], inputs.shape[-1])
            features = backbone(inputs)
            features = features.view(-1, cfg.sequence_length, 2048)
            outputs, _ = model(features)
            scores = torch.nn.functional.softmax(outputs, dim=1)
            roc_est.update(live_scores=scores[labels == alive_lbl, alive_lbl].tolist(),
                                attack_scores=scores[labels != alive_lbl, alive_lbl].tolist())
    eer, err_s = roc_est.estimate_eer()
    return eer


w_e = list(filter(lambda x: 'encoder' in x, os.listdir('weights')))
metrics = pd.DataFrame(columns=list(dataloaders.keys()), index=w_e)

for w in w_e:
    model = torch.load(f'weights/{w}')
    model.to(device)
    backbone = torch.load(f'weights/{w.replace("encoder_", "")}')
    backbone.to(device)
    for data, dataloader in dataloaders.items():
        eer = test(0, dataloader, mode='data', backbone=backbone, model=model)
        metrics.loc[w, data] = eer


print(metrics)
metrics.to_csv('result.csv')