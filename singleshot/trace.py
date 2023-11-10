import torch
import torchvision.models
from easydict import EasyDict
from tools import read_img_as_torch_tensor, check_absolute_difference, benchmark_inference
import os

cfg = EasyDict()
cfg.crop_size = (224, 224)
cfg.batch_size = 4
cfg.num_classes = 2
cfg.alive_lbl = 1
cfg.benchmark = True
cfg.warmup_iters = 5
cfg.work_iters = 30
cfg.model_name = "effnet_v2_s"
crop_format = '256x60x0.1' if cfg.crop_size[0] == 256 else '224x90x0.2'

for key in cfg:
    print(f" - {key}: {cfg[key]}")

# ---------- DNN --------------

if cfg.model_name == "effnet_v2_s":
    model = torchvision.models.efficientnet_v2_s()
    model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=cfg.num_classes, bias=True)
    model.classifier.append(torch.nn.Softmax(dim=1))
else:
    raise NotImplementedError

weights_path = os.path.join(f'./weights/{cfg.model_name}@{crop_format}.pth')
print(f" - model weights: '{weights_path}'")
model.load_state_dict(torch.load(weights_path, map_location='cpu').state_dict())
model.eval()

# ----------------------------

t_fake = read_img_as_torch_tensor(f"./singleshot/resources/{crop_format}/fake.jpg", size=cfg.crop_size)
t_fake = torch.stack(cfg.batch_size * [t_fake])
t_live = read_img_as_torch_tensor(f"./singleshot/resources/{crop_format}/live.jpg", size=cfg.crop_size)
t_live = torch.stack(cfg.batch_size * [t_live])

with torch.no_grad():
    print(" - prediction for the fake sample:", model(t_fake))
    print(" - prediction for the live sample:", model(t_live))

# ----------------------------

traced_model = torch.jit.trace(model, torch.concat([t_fake, t_live]))

target_filename = f"./weights/tmp_{cfg.model_name}@{crop_format}.jit"
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

