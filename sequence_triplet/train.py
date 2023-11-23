import os
import sys
import torch
import torchvision.models
from torch import optim
from torch import nn
from tqdm import tqdm
from collections import Counter
from easydict import EasyDict as edict
from tools import Averagemeter, Speedometer, print_one_line, model_size_mb, ROCEstimator
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp
from encoder import EncoderNet
from copy import deepcopy
import resnet
import loss
from dataset import CustomDataSet, CustomTripletDataSet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------- CONFIG -----------

os.environ['TORCH_HOME'] = './weights'

cfg = edict()
cfg.train_in_fp16 = True
cfg.crop_size = (256, 256)
cfg.sequence_length = 10  # samples to be selected per sequence
cfg.batch_size = 4       # sequences to be selected in minibatch
cfg.grad_accum_batches = 16
cfg.num_epochs = 32
cfg.num_classes = 2
cfg.augment = True
cfg.backbone_name = 'resnext50'#"effnet_v2_s"
cfg.labels_smoothing = 0.1
cfg.max_batches_per_train_epoch = 512 #- use all batches
cfg.triplet_coef = 0.1#1
cfg.ce_coef = 1#0
cfg.triplet_src = 'custom' #'torch'#
crop_format = '256x60x0.1' if cfg.crop_size[0] == 256 else '224x90x0.2'
local_path = f'../sequence/{crop_format}'

for key in cfg:
    print(f" - {key}: {cfg[key]}")

# ---------- SINGLE SHOT BACKBONE --------------

backbone_weights = os.path.join(f'./weights/{cfg.backbone_name}@{crop_format}.pth')
print(f" - backbone weights: '{backbone_weights}'")

if cfg.backbone_name == "effnet_v2_s":
    backbone = torchvision.models.efficientnet_v2_s()
    backbone.classifier[1] = nn.Linear(in_features=1280, out_features=cfg.num_classes, bias=True)
    backbone.load_state_dict(torch.load(backbone_weights).state_dict())
    singleshot = deepcopy(backbone)  # we need copy with last layer to perform naive averaging test
    backbone.classifier[1] = nn.Identity()
    features_size = 1280
elif cfg.backbone_name == "resnet18-ir":
    backbone = model = resnet.ResNetFaceGray(block=resnet.IRBlock, layers=[2, 2, 2, 2], use_se=True, attention=False,
                                             output_features=cfg.num_classes)
    singleshot = deepcopy(backbone)  # we need copy with last layer to perform naive averaging test
    backbone.bn6 = nn.Identity()
    features_size = 512
elif cfg.backbone_name == "resnext50":
    backbone = torch.load(backbone_weights)
    singleshot = deepcopy(backbone)  # we need copy with last layer to perform naive averaging test
    backbone.fc = torch.nn.Identity()
    features_size = 2048
else:
    raise NotImplementedError

backbone.to(device)
backbone.eval()

print(f" - backbone size: {model_size_mb(backbone):.3f} MB")

# -------- SEQUENCE PROCESSING DNN ------------

model = EncoderNet(d_model=features_size, num_heads=8, num_layers=1, d_ff=128, dropout_l=0.1, dropout=0.1,
                   num_classes=cfg.num_classes, max_seq_length=cfg.sequence_length)
model = model.to(device)

print(f" - model size: {model_size_mb(model):.3f} MB")

# Loss and optimizer
loss_fn_tr = loss.TripletLoss() if cfg.triplet_src == 'custom' else nn.TripletMarginLoss()
loss_fn_ce = nn.CrossEntropyLoss(label_smoothing=cfg.labels_smoothing) 
optimizer = optim.Adam([{"params": model.parameters()}, {"params": backbone.parameters()}], lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.2, min_lr=0.00001,
                                                       verbose=True)

# ----------------------------

writer = SummaryWriter(filename_suffix=f'{crop_format}@sequence++')

print("Train dataset:")
train_dataset = CustomTripletDataSet([f'{local_path}/FaceForensics++',
                                      f'{local_path}/roop',
                                      f'{local_path}/dfdc'
                                      ],
                                     tsize=cfg.crop_size,
                                     do_aug=cfg.augment,
                                     min_sequence_length=cfg.sequence_length,
                                     balanced=True)

train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=cfg.batch_size,
                                               drop_last=True, num_workers=8)

alive_lbl = 1

print("val dataset:")
val_dataset = CustomDataSet([
    f"{local_path}/Celeb-DF-v2",
    f"{local_path}/toloka",
    f"{local_path}/toloka_enh",
    f"{local_path}/dfdc_test",

],
    tsize=cfg.crop_size,
    do_aug=False,
    min_sequence_length=cfg.sequence_length)
print(f"  {val_dataset.labels_names()}")
print(f"  {dict(Counter(val_dataset.targets))}")
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True,
                                              drop_last=False, num_workers=6)


metrics = {
    'train': {'EER': float('inf'), 'loss': float('inf'), 'BPCER@0.01': float('inf')},
    'val': {'EER': float('inf'), 'loss': float('inf'), 'BPCER@0.01': float('inf')},
    'test': {'EER': float('inf'), 'loss': float('inf'), 'BPCER@0.01': float('inf')}
}



def update_metrics(mode, epoch, running_loss, roc_estimator):
    if not os.path.exists('./weights'):
        os.makedirs('./weights')
    writer.add_scalar(f"Loss/{mode}", running_loss, epoch)
    if running_loss < metrics[mode]['loss']:
        metrics[mode]['loss'] = running_loss
        print(f" - loss:  {running_loss:.5f} - improvement")
    else:
        print(f" - loss:  {running_loss:.5f}")
    eer, err_s = roc_estimator.estimate_eer()
    writer.add_scalar(f"EER/{mode}", eer, epoch)
    if eer < metrics[mode]['EER']:
        metrics[mode]['EER'] = eer
        if mode == 'val':
            torch.save(model, f"./weights/++tmp_encoder_{cfg.backbone_name}@{crop_format}_{mode}_{round(metrics[mode]['EER'], 2)}_tr{cfg.triplet_coef}_ce{cfg.ce_coef}_{cfg.triplet_src}.pth")
            torch.save(backbone, f"./weights/++tmp_{cfg.backbone_name}@{crop_format}_{mode}_{round(metrics[mode]['EER'], 2)}_tr{cfg.triplet_coef}_ce{cfg.ce_coef}_{cfg.triplet_src}.pth")
        print(f" - EER: {eer:.4f} (score: {err_s:.3f}) - improvement")
    else:
        print(f" - EER: {eer:.4f} (score: {err_s:.3f})")
    print(f" - BPCER@0.1: {roc_estimator.estimate_bpcer(target_apcer=0.1):.4f}")
    bpcer01 = roc_estimator.estimate_bpcer(target_apcer=0.01)
    print(f" - BPCER@0.01: {bpcer01:.4f}")
    writer.add_scalar(f"BPCER@0.01/{mode}", bpcer01, epoch)
    print(f" - BPCER@0.001: {roc_estimator.estimate_bpcer(target_apcer=0.001):.4f}")


loss_avgm = Averagemeter()
loss_tr_avgm = Averagemeter()
loss_ce_avgm = Averagemeter()
ae_avgm = Averagemeter()
speedometer = Speedometer()
scaler = amp.grad_scaler.GradScaler()
train_roc_est = ROCEstimator()
test_roc_est = ROCEstimator()


def train(epoch, dataloader):
    print("TRAIN:")
    train_roc_est.reset()
    loss_avgm.reset()
    loss_tr_avgm.reset()
    loss_ce_avgm.reset()
    ae_avgm.reset()
    speedometer.reset()
    model.train()
    backbone.train()
    running_loss = 0
    true_positive_live = 0
    false_positive_live = 0
    true_negative_live = 0
    false_negative_live = 0
    samples_enrolled = 0
    for batch_idx, (anch, pos, neg, labels) in enumerate(dataloader):
        if batch_idx == cfg.max_batches_per_train_epoch:
            break
        inputs = torch.cat([anch, pos, neg], dim=1).to(device).view(-1, 3, cfg.crop_size[0], cfg.crop_size[1])
        labels = torch.stack([labels, labels, 1 - labels], dim=1).view(-1).to(device)
        inputs = inputs.to(device)
        labels = labels.to(device)
        with amp.autocast(enabled=cfg.train_in_fp16):
            inputs = inputs.view(-1, 3, inputs.shape[-1], inputs.shape[-1])
            features = backbone(inputs)
            features = features.view(cfg.batch_size * 3, cfg.sequence_length, -1)
            outputs, features = model(features)
            loss_ce = loss_fn_ce(outputs, labels)
            if cfg.triplet_src == 'custom':
                loss_tr = loss_fn_tr(features, labels)
            elif cfg.triplet_src == 'torch':
                features = features.view(cfg.batch_size, 3, features_size)
                loss_tr = loss_fn_tr(features[:, 0], features[:, 1], features[:, 2]) 
            loss = cfg.ce_coef * loss_ce + cfg.triplet_coef * loss_tr
            loss /= cfg.grad_accum_batches
        if cfg.train_in_fp16:
            scaler.scale(loss).backward()
            if (batch_idx + 1) % cfg.grad_accum_batches == 0 or batch_idx == (len(dataloader) - 1):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (batch_idx + 1) % cfg.grad_accum_batches == 0 or batch_idx == (len(dataloader) - 1):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                optimizer.zero_grad()
        with torch.no_grad():
            scores = torch.nn.functional.softmax(outputs, dim=1)
            train_roc_est.update(live_scores=scores[labels == alive_lbl, alive_lbl].tolist(),
                                 attack_scores=scores[labels != alive_lbl, alive_lbl].tolist())
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            _tp_live = (predicted[labels == alive_lbl] == alive_lbl).sum().item()
            true_positive_live += _tp_live
            _fp_live = (predicted[labels != alive_lbl] == alive_lbl).sum().item()
            false_positive_live += _fp_live
            _tn_live = (predicted[labels != alive_lbl] != alive_lbl).sum().item()
            true_negative_live += _tn_live
            _fn_live = (predicted[labels == alive_lbl] != alive_lbl).sum().item()
            false_negative_live += _fn_live
            loss_avgm.update(loss.item())
            loss_tr_avgm.update(loss_tr.item())
            loss_ce_avgm.update(loss_ce.item())
            ae_avgm.update((_fp_live / (_fp_live + _tn_live + 1E-6) + _fn_live / (_fn_live + _tp_live + 1E-6)) / 2)
            samples_enrolled += labels.size(0)
            speedometer.update(inputs.size(0))
        print_one_line(
            f'Epoch {epoch} >> loss {loss_avgm.val:.3f}, loss_tr {loss_tr_avgm.val:.3f}, '
            f'loss_ce {loss_ce_avgm.val:.3f}, AE {ae_avgm.val * 100:.2f}%, '
            f'{samples_enrolled}/{len(train_dataset)} ~ '
            f'{100 * samples_enrolled / len(train_dataset):.1f} % | '
            f'{speedometer.speed():.0f} samples / s '
        )
    update_metrics('train', epoch, running_loss / batch_idx, train_roc_est)


def test(epoch, dataloader, mode):
    print(f"{mode.upper()}:")
    test_roc_est.reset()
    model.eval()
    backbone.eval()
    running_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, file=sys.stdout)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.view(-1, 3, inputs.shape[-1], inputs.shape[-1])
            features = backbone(inputs)
            features = features.view(-1, cfg.sequence_length, features_size)
            outputs, _ = model(features)
            loss = loss_fn_ce(outputs, labels)
            running_loss += loss.item()
            scores = torch.nn.functional.softmax(outputs, dim=1)
            test_roc_est.update(live_scores=scores[labels == alive_lbl, alive_lbl].tolist(),
                                attack_scores=scores[labels != alive_lbl, alive_lbl].tolist())
    update_metrics(mode, epoch, running_loss / len(dataloader), test_roc_est)
    if mode == 'val':
        scheduler.step(metrics[mode]['EER'])
    print("\n")


def test_naive_averaging(dataloader):
    print("\nNAIVE AVERAGING TEST:")
    singleshot.to(device)
    singleshot.eval()
    test_roc_est.reset()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, file=sys.stdout)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.view(-1, 3, inputs.shape[-1], inputs.shape[-1])
            features = singleshot(inputs)
            features = features.view(len(inputs)// cfg.sequence_length, cfg.sequence_length, -1)
            scores = torch.nn.functional.softmax(features, dim=2)[:, :, alive_lbl].mean(dim=1)
            test_roc_est.update(live_scores=scores[labels == alive_lbl].tolist(),
                                attack_scores=scores[labels != alive_lbl].tolist())
    eer, err_s = test_roc_est.estimate_eer()
    print(f" - EER: {eer:.4f} (score: {err_s:.4f})")
    print(f" - BPCER@0.1: {test_roc_est.estimate_bpcer(target_apcer=0.1):.4f}")
    print(f" - BPCER@0.01: {test_roc_est.estimate_bpcer(target_apcer=0.01):.4f}")
    print(f" - BPCER@0.001: {test_roc_est.estimate_bpcer(target_apcer=0.001):.4f}")


if __name__ == '__main__':
    test_naive_averaging(val_dataloader)
    for epoch in range(cfg.num_epochs):
        train(epoch, train_dataloader)
        test(epoch, val_dataloader, 'val')
        train_dataloader.dataset.generate_triplets()

