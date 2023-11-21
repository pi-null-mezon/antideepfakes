import json
from shutil import move
import os
from tqdm import tqdm

source_path = '/media/alex/HDD1_2T/Deepfakes/dfdc/train_part_33/dfdc_train_part_33'

with open(os.path.join(source_path, 'metadata.json'), 'r') as i_f:
    markup = json.load(i_f)

target_path = source_path.rsplit('/', 1)[0]
os.makedirs(os.path.join(target_path, 'live'), exist_ok=True)
os.makedirs(os.path.join(target_path, 'fake'), exist_ok=True)
for filename in tqdm(markup):
    label = 'live' if markup[filename]['label'] == 'REAL' else 'fake'
    src = os.path.join(source_path, filename)
    target = os.path.join(target_path, label, filename)
    move(src, target)
