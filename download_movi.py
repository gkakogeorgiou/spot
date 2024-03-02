# Copied and modified from https://github.com/singhgautam/steve/blob/evaluate/download_movi_with_masks.py
import os
import torch
import argparse
from tqdm import tqdm
import tensorflow_datasets as tfds
import torchvision.utils as vutils

from torchvision import transforms


parser = argparse.ArgumentParser()

parser.add_argument('--out_path', default='MOVi/')
parser.add_argument('--level', default='e', help='c or e')
parser.add_argument('--split', default='train', help='train, validation or test')

parser.add_argument('--version', default='1.0.0')
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--max_num_objs', type=int, default=25)

args = parser.parse_args()

ds, ds_info = tfds.load(f"movi_{args.level}/{args.image_size}x{args.image_size}:{args.version}", data_dir="gs://kubric-public/tfds", with_info=True)
train_iter = iter(tfds.as_numpy(ds[args.split]))

to_tensor = transforms.ToTensor()

b = 0
print('Please be patient; it is usually very slow.')
for record in tqdm(train_iter):
    video = record['video']
    if args.split != 'train':
        masks = record["segmentations"]
    T, *_ = video.shape

    # setup dirs
    path_vid = os.path.join(args.out_path, args.level, args.split, f"{b:08}")
    os.makedirs(path_vid, exist_ok=True)

    for t in range(T):
        img = video[t]
        img = to_tensor(img)
        vutils.save_image(img, os.path.join(path_vid, f"{t:08}_image.png"))
        if args.split != 'train':
            for n in range(args.max_num_objs):
                mask = (masks[t] == n).astype(float)
                mask = torch.Tensor(mask).permute(2, 0, 1)
                vutils.save_image(mask, os.path.join(path_vid, f'{t:08}_mask_{n:02}.png'))

    b += 1
