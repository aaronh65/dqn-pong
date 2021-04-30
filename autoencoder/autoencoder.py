from datetime import datetime
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from models import ResnetEncoder, ResnetDecoder
from dataloader import get_dataloader
from utils import spatial_norm

# import wandb
import cv2
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import argparse

DISPLAY=True

class AutoEncoder(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.res_encoder = ResnetEncoder()
        self.res_decoder = ResnetDecoder()
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, rgb, decode=True):
        latent = self.res_encoder(rgb)
        if decode:
            pred_masks = self.res_decoder(latent)
            return pred_masks, latent
        else:
            return latent

    def training_step(self, batch, batch_nb):
        rgb = batch['rgb']

        skier = batch['skier']
        flags = batch['flags']
        rocks = batch['rocks']
        trees = batch['trees']

        gt_masks = torch.cat((skier, flags, rocks, trees), dim=1)
        latent = self.res_encoder(rgb)
        pred_masks = self.res_decoder(latent)

        class_loss = self.criterion(pred_masks, gt_masks) # N,C,H,W
        class_loss = class_loss.sum((-1,-2))

        metrics = {}
        metrics['train/loss'] = class_loss.mean().item()

        if self.global_step % 50 == 0:
            visuals = self.make_visuals(rgb, gt_masks, pred_masks)
            metrics['train_image'] = wandb.Image(visuals)
        if self.logger is not None:
            self.logger.log_metrics(metrics, self.global_step)
        loss = class_loss.mean((-1,-2), keepdim=True).squeeze(-1)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        rgb = batch['rgb']

        skier = batch['skier']
        flags = batch['flags']
        rocks = batch['rocks']
        trees = batch['trees']

        gt_masks = torch.cat((skier, flags, rocks, trees), dim=1)
        latent = self.res_encoder(rgb)
        pred_masks = self.res_decoder(latent)

        class_loss = self.criterion(pred_masks, gt_masks)
        class_loss = class_loss.sum((-1,-2))
        metrics = {}
        metrics['val/loss'] = class_loss.mean().item()
        if batch_nb == 0:
            visuals = self.make_visuals(rgb, gt_masks, pred_masks)
            metrics['val_image'] = wandb.Image(visuals)
        if self.logger is not None:
            self.logger.log_metrics(metrics, self.global_step)
        val_loss = class_loss.mean((-1,-2), keepdim=True).squeeze(-1)
        return {'val_loss': val_loss}

    def validation_epoch_end(self, batch_metrics):
        results = dict()

        for metrics in batch_metrics:
            for key in metrics:
                if key not in results:
                    results[key] = list()
                results[key].append(metrics[key].mean().item())

        summary = {key: np.mean(val) for key, val in results.items()}
        if self.logger != None:
            self.logger.log_metrics(summary, self.global_step)
        return summary

    def train_dataloader(self):
        return get_dataloader(self.hparams, is_train=True)

    def val_dataloader(self):
        return get_dataloader(self.hparams, is_train=False)

    def configure_optimizers(self):
        # add in lr from hparams if default adam sucks
        optim = torch.optim.Adam(list(self.res_encoder.parameters()) + list(self.res_decoder.parameters())) 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=2, min_lr=1e-6, verbose=True)
        return [optim], [scheduler]

    def make_visuals(self, rgb, gt_masks, pred_masks, num_samples=2):
        all_images = list()
        rgb = rgb.cpu().numpy().transpose(0,2,3,1)
        gt_masks = gt_masks.detach().cpu().numpy()
        pred_masks = spatial_norm(pred_masks).detach().cpu().numpy()

        for b in range(num_samples):
            images = list()
            num_classes = gt_masks.shape[1]

            # normalize and convert to numpy
            
            for c in range(num_classes):

                _rgb = np.uint8(rgb[b]*255) # H,W,3
                _gt = gt_masks[b,c:c+1].transpose(1,2,0) # H,W,1
                _gt = np.uint8(_gt*255) # H,W,1
                _gt = np.tile(_gt, (1,1,3)) # H,W,3

                _pred = pred_masks[b,c:c+1].transpose(1,2,0) # H,W,1
                _pred = np.uint8(_pred*255) # H,W,1
                _pred = np.tile(_pred, (1,1,3)) # H,W,3

                _combined = np.concatenate((_rgb, _gt, _pred), axis=1) # H,3W,3
                images.append(_combined)

            images = np.vstack(images) # 8 by 3 images each row H,3W,3
            all_images.append(images)
        all_images = np.hstack(all_images)
        all_images = cv2.cvtColor(all_images, cv2.COLOR_RGB2BGR)
        if DISPLAY:
            cv2.imshow('debug', all_images)
            cv2.waitKey(1000)
        return all_images

def main(hparams):
    if hparams.log:
        logger = WandbLogger(save_dir=hparams.save_dir, project='vlr-project')
    else:
        logger = False

    checkpoint_callback = ModelCheckpoint(hparams.save_dir, save_top_k=3)
    model = AutoEncoder(hparams)
    trainer = pl.Trainer(
            max_epochs=hparams.max_epochs,
            gpus=hparams.gpus,
            checkpoint_callback=checkpoint_callback,
            enable_pl_optimizer=False,
            logger=logger,
            distributed_backend='dp'
            )
    trainer.fit(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='data/20210413_182405')
    parser.add_argument('--gpus', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=2)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--log', action='store_true')
    args = parser.parse_args()

    save_dir = Path(args.save_dir) / 'autoencoder' / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(exist_ok=True, parents=True)
    args.save_dir = str(save_dir)
    main(args)
