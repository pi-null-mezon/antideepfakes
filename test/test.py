import torch
from easydict import EasyDict
from tools import CustomDataSet, ROCEstimator
from collections import Counter
from tqdm import tqdm
import sys

cfg = EasyDict()
cfg.train_in_fp16 = True
cfg.crop_size = (256, 256)
cfg.sequence_length = 10  # samples to be selected per sequence
cfg.batch_size = 32  # sequences to be selected in minibatch
cfg.augment = False
cfg.backbone_name = "effnet_v2_s"
crop_format = '256x60x0.1' if cfg.crop_size[0] == 256 else '224x90x0.2'
local_path = f"/home/alex/Fastdata/deepfakes/sequence/{crop_format}"

alive_lbl = 1
device = torch.device('cuda')

for key in cfg:
    print(f" - {key}: {cfg[key]}")

# --------------------------------

singleshot_model = torch.load(f'./weights/{cfg.backbone_name}@{crop_format}.pth').to(device)
singleshot_model.classifier.append(torch.nn.Softmax(dim=1))
singleshot_model.eval()

traced_singleshot_model = torch.jit.load(f'./weights/tmp_{cfg.backbone_name}@{crop_format}.jit').to(device)
traced_singleshot_model.eval()

sequence_model = torch.jit.load(f'./weights/tmp_dd_on_{cfg.backbone_name}@{crop_format}.jit').to(device)
sequence_model.eval()

# ---------------------------------

test_dataset = CustomDataSet([
    f"{local_path}/dfdc/train_part_17",
    f"{local_path}/dfdc/train_part_41"
],
    tsize=cfg.crop_size,
    do_aug=cfg.augment,
    min_sequence_length=cfg.sequence_length)
print(f"  {test_dataset.labels_names()}")
print(f"  {dict(Counter(test_dataset.targets))}")
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True,
                                              drop_last=True, num_workers=6)


def test_naive_averaging_singleshot(dataloader, singleshot, info):
    print(f"\nNAIVE AVERAGING TEST for {info}")
    test_roc_est = ROCEstimator()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, file=sys.stdout)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.view(-1, inputs.shape[-3], inputs.shape[-2], inputs.shape[-1])
            scores = singleshot(inputs).view(cfg.batch_size, cfg.sequence_length, -1)
            scores = scores[:, :, alive_lbl].mean(dim=1)
            test_roc_est.update(live_scores=scores[labels == alive_lbl].tolist(),
                                attack_scores=scores[labels != alive_lbl].tolist())
    eer, err_s = test_roc_est.estimate_eer()
    print(f" - EER: {eer:.4f} (score: {err_s:.4f})")
    print(f" - BPCER@0.1: {test_roc_est.estimate_bpcer(target_apcer=0.1):.4f}")
    print(f" - BPCER@0.01: {test_roc_est.estimate_bpcer(target_apcer=0.01):.4f}")
    print(f" - BPCER@0.001: {test_roc_est.estimate_bpcer(target_apcer=0.001):.4f}")


def test_sequence_processor(dataloader, processor, info):
    print(f"\nSEQUENCE PROCESSOR TEST for {info}")
    test_roc_est = ROCEstimator()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, file=sys.stdout)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            scores = processor(inputs)[:, alive_lbl]
            test_roc_est.update(live_scores=scores[labels == alive_lbl].tolist(),
                                attack_scores=scores[labels != alive_lbl].tolist())
    eer, err_s = test_roc_est.estimate_eer()
    print(f" - EER: {eer:.4f} (score: {err_s:.4f})")
    print(f" - BPCER@0.1: {test_roc_est.estimate_bpcer(target_apcer=0.1):.4f}")
    print(f" - BPCER@0.01: {test_roc_est.estimate_bpcer(target_apcer=0.01):.4f}")
    print(f" - BPCER@0.001: {test_roc_est.estimate_bpcer(target_apcer=0.001):.4f}")

info = f"sequence length: {cfg.sequence_length} frames + "

test_sequence_processor(dataloader=test_dataloader, processor=sequence_model, info=info + f"encoder@{cfg.backbone_name}@{crop_format}.jit")
#test_naive_averaging_singleshot(dataloader=test_dataloader, singleshot=singleshot_model, info=info + f"{cfg.backbone_name}@{crop_format}.pth")
test_naive_averaging_singleshot(dataloader=test_dataloader, singleshot=traced_singleshot_model, info=info + f"{cfg.backbone_name}@{crop_format}.jit")