import argparse
import numpy as np
import cv2
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from atari_masks import get_hsv_colors, get_mask

from matplotlib import pyplot as plt

DEBUG = False

class SkierDataset(Dataset):
    def __init__(self, episodes, env):
        self.frames = list()
        self.env = env
        for ep_path in episodes:
            rgb_path = ep_path / 'rgb'
            self.frames.extend(sorted(list(rgb_path.glob('*'))))
        self.classes_hsv = get_hsv_colors(env) # object to hsv color dict

        self.transform = transforms.Compose([
            #transforms.Resize((256, 160)),
            transforms.Resize((88, 88)),
            transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        res = {} # a dictionary of tensors

        frame = self.frames[i]
        rgb = np.array(Image.open(frame))
        
        h,w,c = rgb.shape

        masks = list()
        for cls, colors in self.classes_hsv.items():
            mask = np.zeros((h,w)).astype(bool)
            for color in colors:
                temp_mask = get_mask(rgb.copy(), color)
                temp_mask = temp_mask.astype(bool)
                mask = np.uint8(np.logical_or(mask, temp_mask))*255
            # block out the top part of the masks so that ball can be reconstructed
            if self.env == 'PongNoFrameskip-v4':
                mask[:h//6] = 0
                mask[9:h//10:] = 0
            mask_image = Image.fromarray(mask)
            mask_tensor = self.transform(mask_image)
            mask_tensor[mask_tensor>0.1] = 1
            res[cls] = mask_tensor
            masks.append(mask)
        masks = np.hstack(masks)

        rgb_tensor = self.transform(Image.fromarray(rgb))
        res['rgb']  = rgb_tensor

        if DEBUG:
            cv2.imshow('masks', masks)
            cv2.waitKey(0)
            cv2.imwrite('rgb.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            rgb[:h//6] = 50
            rgb[9*h//10:] = 50
            cv2.imwrite('rgb_zero.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite('masks.png', masks)

        return res

def get_dataloader(args, is_train=False):
    data_dir = Path(args.dataset_dir)

    paths = sorted(list(data_dir.glob('*')))

    episodes = list()
    for i, path in enumerate(paths):
        if is_train and i % 10 != 0:
            episodes.append(path)
        elif not is_train and i % 10 == 0:
            episodes.append(path)

    dataset = SkierDataset(episodes, args.env)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    return dataloader
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='data/20210413_182405')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    
    dataloader = get_dataloader(args, is_train=True)
    
    print(f'with batch_size={args.batch_size}, train dataloader has {len(dataloader)} batches')
    for batch_nb, batch in enumerate(dataloader):
        rgb = batch['rgb']
        skier = batch['skier']

        print(rgb.shape)
        print(skier.shape)

        break



