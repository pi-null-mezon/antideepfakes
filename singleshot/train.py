import os
import sys
import torch
import torchvision.models
from torch import optim
from torch import nn
from tqdm import tqdm
from collections import Counter
from tools import CustomDataSet, Avgmeter, Speedometer, print_one_line
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp
import resnet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------- CONFIG -----------

os.environ['TORCH_HOME'] = './weights'
train_in_fp16 = True
crop_size = (224, 224)
batch_size = 32
grad_accum_batches = 4
num_epochs = 50
num_classes = 2
augment = True
model_name = "resnet18-ir"
pretrained = True
labels_smoothing = 0.1

crop_format = '256x60x0.1' if crop_size[0] == 256 else '224x90x0.2'
local_path = f"/home/alex/Fastdata/deepfakes/singleshot/{crop_format}"

# ---------- DNN --------------
if model_name == "effnet_v2_s":
    model = torchvision.models.efficientnet_v2_s()
    if pretrained:
        model.load_state_dict(torchvision.models.efficientnet_v2_s(
            weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1).state_dict())
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
elif model_name == "resnet18-ir":
    model = resnet.ResNetFaceGray(block=resnet.IRBlock, layers=[2, 2, 2, 2], use_se=True, attention=False,
                                  output_features=num_classes)

model = model.to(device)
# model = nn.DataParallel(model)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss(label_smoothing=labels_smoothing)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.1, min_lr=0.00001,
                                                       verbose=True)

# ----------------------------

writer = SummaryWriter(filename_suffix=model_name)

print("Train dataset:")
train_dataset = CustomDataSet([f"{local_path}/FaceForensics++",
                               f"{local_path}/Celeb-DF-v2",
                               f"{local_path}/dfdc/train_part_0",
                               f"{local_path}/dfdc/train_part_1",
                               f"{local_path}/dfdc/train_part_3",
                               f"{local_path}/dfdc/train_part_5",
                               f"{local_path}/dfdc/train_part_07",
                               f"{local_path}/dfdc/train_part_11",
                               f"{local_path}/dfdc/train_part_23",
                               f"{local_path}/dfdc/train_part_43",
                               f"{local_path}/dfdc/train_part_47"], crop_size, do_aug=augment)
print(f"  {train_dataset.labels_names()}")
assert num_classes == len(train_dataset.labels_names())
lbls_count = dict(Counter(train_dataset.targets))
print(f"  {lbls_count}")
class_weights = list(1 / torch.Tensor(list(lbls_count.values())))
samples_weights = [class_weights[lbl] for lbl in train_dataset.targets]
sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weights, num_samples=len(train_dataset),
                                                 replacement=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=sampler, batch_size=batch_size, num_workers=8)

alive_lbl = None
for key in train_dataset.labels_names():
    if train_dataset.labels_names()[key] == 'live':
        alive_lbl = key
        break

print("Test dataset:")
test_dataset = CustomDataSet([f"{local_path}/dfdc/train_part_17",
                              f"{local_path}/dfdc/train_part_41"], crop_size, do_aug=False)
print(f"  {test_dataset.labels_names()}")
print(f"  {dict(Counter(test_dataset.targets))}")
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=6)

metrics = {
    'train': {'BPCER': float('inf'), 'APCER': float('inf'), 'loss': float('inf'), 'AE': float('inf')},
    'test': {'BPCER': float('inf'), 'APCER': float('inf'), 'loss': float('inf'), 'AE': float('inf')}
}


def update_metrics(mode, epoch,
                   running_loss,
                   true_positive_live,
                   false_positive_live,
                   true_negative_live,
                   false_negative_live):
    print(f"{mode.upper()}:")
    if not os.path.exists('./weights'):
        os.makedirs('./weights')
    writer.add_scalar(f"Loss/{mode}", running_loss, epoch)
    if running_loss < metrics[mode]['loss']:
        metrics[mode]['loss'] = running_loss
        print(f" - loss:  {running_loss:.5f} - improvement")
    else:
        print(f" - loss:  {running_loss:.5f}")
    prob = false_positive_live / (false_positive_live + true_negative_live + 1E-6)
    writer.add_scalar(f"APCER/{mode}", prob, epoch)
    if prob < metrics[mode]['APCER']:
        metrics[mode]['APCER'] = prob
        print(f" - APCER: {prob:.5f} - improvement")
    else:
        print(f" - APCER: {prob:.5f}")
    apcer = prob
    prob = false_negative_live / (false_negative_live + true_positive_live + 1E-6)
    writer.add_scalar(f"BPCER/{mode}", prob, epoch)
    if prob < metrics[mode]['BPCER']:
        metrics[mode]['BPCER'] = prob
        print(f" - BPCER: {prob:.5f} - improvement")
    else:
        print(f" - BPCER: {prob:.5f}")
    bpcer = prob
    prob = (bpcer + apcer) / 2
    writer.add_scalar(f"Average error/{mode}", prob, epoch)
    if prob < metrics[mode]['AE']:
        metrics[mode]['AE'] = prob
        if mode == 'test':
            torch.save(model, f"./weights/{model_name}@{crop_format}.pth")
        print(f" - Average error: {prob:.5f} - improvement")
    else:
        print(f" - Average error: {prob:.5f}")


loss_avgm = Avgmeter()
ae_avgm = Avgmeter()
speedometer = Speedometer()
scaler = amp.grad_scaler.GradScaler()


def train(epoch, dataloader):
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
        inputs = inputs.to(device)
        labels = labels.to(device)
        if train_in_fp16:
            with amp.autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss = loss / grad_accum_batches
            scaler.scale(loss).backward()
            if (batch_idx + 1) % grad_accum_batches == 0 or batch_idx == (len(dataloader) - 1):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss = loss / grad_accum_batches
            loss.backward()
            if (batch_idx + 1) % grad_accum_batches == 0 or batch_idx == (len(dataloader) - 1):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                optimizer.zero_grad()
        with torch.no_grad():
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
            f'Epoch {epoch} >> loss {loss_avgm.avg:.3f}, AE {ae_avgm.avg * 100:.2f}%, '
            f'{samples_enrolled}/{len(train_dataset)} ~ '
            f'{100 * samples_enrolled / len(train_dataset):.1f} % | '
            f'{speedometer.speed():.0f} samples / s '
        )
    update_metrics('train', epoch,
                   running_loss / len(dataloader),
                   true_positive_live,
                   false_positive_live,
                   true_negative_live,
                   false_negative_live)


def test(epoch, dataloader):
    model.eval()
    running_loss = 0
    true_positive_live = 0
    false_positive_live = 0
    true_negative_live = 0
    false_negative_live = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, file=sys.stdout)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            true_positive_live += (predicted[labels == alive_lbl] == alive_lbl).sum().item()
            false_positive_live += (predicted[labels != alive_lbl] == alive_lbl).sum().item()
            true_negative_live += (predicted[labels != alive_lbl] != alive_lbl).sum().item()
            false_negative_live += (predicted[labels == alive_lbl] != alive_lbl).sum().item()
    update_metrics('test', epoch,
                   running_loss / len(dataloader),
                   true_positive_live,
                   false_positive_live,
                   true_negative_live,
                   false_negative_live)
    scheduler.step(metrics['test']['AE'])


for epoch in range(num_epochs):
    train(epoch, train_dataloader)
    test(epoch, test_dataloader)
