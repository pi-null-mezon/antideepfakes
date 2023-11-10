import os
import sys
import torch
import torchvision.models
from torch import optim
from torch import nn
from tqdm import tqdm
from collections import Counter
from easydict import EasyDict as edict
from tools import CustomDataSet, Averagemeter, Speedometer, print_one_line, model_size_mb, ROCEstimator
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp
import resnet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------- CONFIG -----------

os.environ['TORCH_HOME'] = './weights'

cfg = edict()
cfg.train_in_fp16 = True
cfg.crop_size = (256, 256)
cfg.batch_size = 32
cfg.grad_accum_batches = 4
cfg.num_epochs = 50
cfg.num_classes = 2
cfg.augment = True
cfg.model_name = "effnet_v2_s"
cfg.pretrained = True
cfg.labels_smoothing = 0.1
cfg.max_batches_per_train_epoch = -1  # -1 - all
crop_format = '256x60x0.1' if cfg.crop_size[0] == 256 else '224x90x0.2'
local_path = f"/home/alex/Fastdata/deepfakes/singleshot/{crop_format}"

for key in cfg:
    print(f" - {key}: {cfg[key]}")

# ---------- DNN --------------

if cfg.model_name == "effnet_v2_s":
    model = torchvision.models.efficientnet_v2_s()
    if cfg.pretrained:
        model.load_state_dict(torchvision.models.efficientnet_v2_s(
            weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1).state_dict())
    model.classifier[1] = nn.Linear(in_features=1280, out_features=cfg.num_classes, bias=True)
elif cfg.model_name == "resnet18-ir":
    model = resnet.ResNetFaceGray(block=resnet.IRBlock, layers=[2, 2, 2, 2], use_se=True, attention=False,
                                  output_features=cfg.num_classes)

model = model.to(device)
# model = nn.DataParallel(model)

print(f" - model size: {model_size_mb(model):.3f} MB")

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.labels_smoothing)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, min_lr=0.00001,
                                                       verbose=True)

# ----------------------------

writer = SummaryWriter(filename_suffix=cfg.model_name)

print("Train dataset:")
train_dataset = CustomDataSet([
    f"{local_path}/FaceForensics++",
    f"{local_path}/Celeb-DF-v2",
    f"{local_path}/dfdc/train_part_0",
    f"{local_path}/dfdc/train_part_1",
    f"{local_path}/dfdc/train_part_3",
    f"{local_path}/dfdc/train_part_5",
    f"{local_path}/dfdc/train_part_07",
    f"{local_path}/dfdc/train_part_11",
    f"{local_path}/dfdc/train_part_23",
    f"{local_path}/dfdc/train_part_43",
    f"{local_path}/dfdc/train_part_47"
], cfg.crop_size, do_aug=cfg.augment)
print(f"  {train_dataset.labels_names()}")
assert cfg.num_classes == len(train_dataset.labels_names())
lbls_count = dict(Counter(train_dataset.targets))
print(f"  {lbls_count}")
class_weights = list(1 / torch.Tensor(list(lbls_count.values())))
samples_weights = [class_weights[lbl] for lbl in train_dataset.targets]
sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weights, num_samples=len(train_dataset),
                                                 replacement=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=sampler, batch_size=cfg.batch_size, num_workers=8)

alive_lbl = None
for key in train_dataset.labels_names():
    if train_dataset.labels_names()[key] == 'live':
        alive_lbl = key
        break

print("Test dataset:")
test_dataset = CustomDataSet([
    f"{local_path}/dfdc/train_part_17",
    f"{local_path}/dfdc/train_part_41",
], cfg.crop_size, do_aug=False)
print(f"  {test_dataset.labels_names()}")
print(f"  {dict(Counter(test_dataset.targets))}")
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=6)

metrics = {
    'train': {'EER': float('inf'), 'loss': float('inf'), 'BPCER@0.01': float('inf')},
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
        if mode == 'test':
            torch.save(model, f"./weights/tmp_{cfg.backbone_name}@{crop_format}.pth")
        print(f" - EER: {eer:.4f} (score: {err_s:.3f}) - improvement")
    else:
        print(f" - EER: {eer:.4f} (score: {err_s:.3f})")
    print(f" - BPCER@0.1: {roc_estimator.estimate_bpcer(target_apcer=0.1):.4f}")
    bpcer01 = roc_estimator.estimate_bpcer(target_apcer=0.01)
    print(f" - BPCER@0.01: {bpcer01:.4f}")
    writer.add_scalar(f"BPCER@0.01/{mode}", bpcer01, epoch)
    print(f" - BPCER@0.001: {roc_estimator.estimate_bpcer(target_apcer=0.001):.4f}")


loss_avgm = Averagemeter()
ae_avgm = Averagemeter()
speedometer = Speedometer()
scaler = amp.grad_scaler.GradScaler()
train_roc_est = ROCEstimator()
test_roc_est = ROCEstimator()


def train(epoch, dataloader):
    print("TRAIN:")
    train_roc_est.reset()
    loss_avgm.reset()
    ae_avgm.reset()
    speedometer.reset()
    model.train()
    running_loss = 0
    true_positive_live = 0
    false_positive_live = 0
    true_negative_live = 0
    false_negative_live = 0
    samples_enrolled = 0
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        if batch_idx == cfg.max_batches_per_train_epoch:
            break
        inputs = inputs.to(device)
        labels = labels.to(device)
        if cfg.train_in_fp16:
            with amp.autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss = loss / cfg.grad_accum_batches
            scaler.scale(loss).backward()
            if (batch_idx + 1) % cfg.grad_accum_batches == 0 or batch_idx == (len(dataloader) - 1):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss = loss / cfg.grad_accum_batches
            loss.backward()
            if (batch_idx + 1) % cfg.grad_accum_batches == 0 or batch_idx == (len(dataloader) - 1):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
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
            ae_avgm.update((_fp_live / (_fp_live + _tn_live + 1E-6) + _fn_live / (_fn_live + _tp_live + 1E-6)) / 2)
            samples_enrolled += labels.size(0)
            speedometer.update(labels.size(0))
        print_one_line(
            f'Epoch {epoch} >> loss {loss_avgm.val:.3f}, AE {ae_avgm.val * 100:.2f}%, '
            f'{samples_enrolled}/{len(train_dataset)} ~ '
            f'{100 * samples_enrolled / len(train_dataset):.1f} % | '
            f'{speedometer.speed():.0f} samples / s '
        )
    update_metrics('train', epoch, running_loss / batch_idx, train_roc_est)


def test(epoch, dataloader):
    print("TEST:")
    test_roc_est.reset()
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, file=sys.stdout)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            scores = torch.nn.functional.softmax(outputs, dim=1)
            test_roc_est.update(live_scores=scores[labels == alive_lbl, alive_lbl].tolist(),
                                attack_scores=scores[labels != alive_lbl, alive_lbl].tolist())
    update_metrics('test', epoch, running_loss / len(dataloader), test_roc_est)
    scheduler.step(metrics['test']['EER'])
    print("\n")


for epoch in range(cfg.num_epochs):
    train(epoch, train_dataloader)
    test(epoch, test_dataloader)
