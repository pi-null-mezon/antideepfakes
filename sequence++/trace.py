import torch
import torchvision.models
from easydict import EasyDict
from tools import read_img_as_torch_tensor, check_absolute_difference, benchmark_inference
from encoder import EncoderNet, DeepfakeDetector
import os

cfg = EasyDict()
cfg.crop_size = (256, 256)
cfg.sequence_length = 10
cfg.batch_size = 4
cfg.num_classes = 2
cfg.alive_lbl = 1
cfg.benchmark = False
cfg.warmup_iters = 5
cfg.work_iters = 30
cfg.model_name = "resnext50"
cfg.pair_path = 'katia_3'
crop_format = '256x60x0.1' if cfg.crop_size[0] == 256 else '224x90x0.2'

for key in cfg:
    print(f" - {key}: {cfg[key]}")

# ---------- DNN --------------

weights_path = os.path.join(f'./weights/{cfg.pair_path}/++tmp_{cfg.model_name}@{crop_format}.pth')
print(f" - single shot weights: '{weights_path}'")
singleshot = torch.load(weights_path, map_location='cpu')
weights_path = os.path.join(f'./weights/{cfg.pair_path}/++tmp_encoder_{cfg.model_name}@{crop_format}.pth')
print(f" - encoder weights: '{weights_path}'")
encoder = torch.load(weights_path, map_location='cpu')

model = DeepfakeDetector(singleshot, encoder)
model.eval()

# ----------------------------

t_fake = read_img_as_torch_tensor(f"./singleshot/resources/{crop_format}/fake.jpg", size=cfg.crop_size)
t_fake = torch.stack(cfg.sequence_length * [t_fake])
t_fake = torch.stack(cfg.batch_size * [t_fake])
t_live = read_img_as_torch_tensor(f"./singleshot/resources/{crop_format}/live.jpg", size=cfg.crop_size)
t_live = torch.stack(cfg.sequence_length * [t_live])
t_live = torch.stack(cfg.batch_size * [t_live])

with torch.no_grad():
    print(" - prediction for the fake sample:", model(t_fake))
    print(" - prediction for the live sample:", model(t_live))

# ----------------------------

traced_model = torch.jit.trace(model, torch.concat([t_fake, t_live]))

target_filename = f"./weights/tmp_dd_on_{cfg.model_name}@{crop_format}.jit"
torch.jit.save(traced_model, target_filename)
del traced_model
traced_model = torch.jit.load(target_filename)

with torch.no_grad():
    for t in [t_fake, t_live]:
        model_prediction = model(t)
        traced_model_prediction = traced_model(t)
        if not check_absolute_difference(model_prediction, traced_model_prediction):
            print("WARNING - high absolute difference:")
            print(model_prediction)
            print(traced_model_prediction)
            print("abort...")
            exit()

print(f"SUCCESS - the model has been successfully traced and saved in '{target_filename}'")

# ----------- BENCHMARK INFERENCE TIME -------------
if cfg.benchmark:
    for device in ['cpu', 'cuda']:
        benchmark_inference(model, device, t_live.shape, cfg.warmup_iters, cfg.work_iters, '.pth')
        benchmark_inference(traced_model, device, t_live.shape, cfg.warmup_iters, cfg.work_iters, '.jit')

