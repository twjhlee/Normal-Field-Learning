from __future__ import print_function
from email.policy import default
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataset.dataset import RGB2NormalDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from models.model import Encoder, Decoder
from losses.loss import get_edge_loss, FocalLoss
import argparse
import cv2
import json
import random
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_loss(pred_norm, gt_norm, gt_norm_mask):
    dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1)
    valid_mask = gt_norm_mask[:, :, :].float() \
                    * (dot.detach() < 0.999).float() \
                    * (dot.detach() > -0.999).float()
    valid_mask = valid_mask > 0.0

    al = torch.acos(dot[valid_mask])
    loss = torch.mean(al)
    return loss

def iterate(train_loader, val_loader, device, writer, config, Encoder, Decoder):
    target_losses = []
    geom_losses = []
    for idx, data in enumerate(train_loader):
        with torch.no_grad():
            rgb = data['rgb'].to(device).squeeze().contiguous()
            target = data['target'][:, :3]
            gt_norm_mask = data['target'][:, -1]
            gt_norm_mask = gt_norm_mask.to(device)
            target = target.to(device).contiguous()
            if target.ndim == 3:
                target = target.unsqueeze(1)

        B = rgb.shape[0]
        z, ll_feat = Encoder(rgb)
        target_pred = Decoder(z, ll_feat, rgb.size()[2:])
        optim.zero_grad()
        if "normal" == config['output_mod']:
            target_loss = get_loss(target_pred, target, gt_norm_mask).squeeze()
        else:
            target_loss = criterion(target_pred, target)
        # norm = torch.linalg.norm(target_pred, dim=1)
        # ones = torch.ones_like(norm)
        # geom_loss = criterion(norm, ones).squeeze()
        # total_loss = target_loss + geom_loss
        target_losses.append(target_loss.item())
        # geom_losses.append(geom_loss.item())
        target_loss.backward()
        optim.step()

    with torch.no_grad():
        # Losses and input images
        losses = np.array(target_losses)
        writer.add_scalar("Train Target Loss", np.mean(losses), epoch)
        # losses = np.array(geom_losses)
        # writer.add_scalar("Train Geom Loss", np.mean(losses), epoch)

        # Targets - depth or normal 
        if len(target) > config['log_img_cnt']:
            rgb = rgb[: config['log_img_cnt']].cpu()
            target = target[: config['log_img_cnt']].cpu()
            target_pred = target_pred[: config['log_img_cnt']].cpu()

        else:
            rgb = rgb.cpu()
            target = target.cpu()
            target_pred = target_pred.cpu()
 
        writer.add_image("Train RGB", torch.clamp(torch.cat(torch.unbind(rgb, dim=0), dim=2), 0, 1), epoch)
        if config['output_mod'] == "normal":
            target = unbind_batch(target)
            target += 1
            target /= 2
            target_pred = unbind_batch(target_pred)
            target_pred += 1
            target_pred /= 2
            target_pred = torch.clip(target_pred, 0, 1)
            writer.add_image("Train GT Normal", target, epoch)
            writer.add_image("Train Pred Normal", target_pred, epoch)

        elif config['output_mod'] == "depth":
            target = unbind_batch(target)
            target_pred = torch.clip(target_pred, 0, 1)
            target_pred = unbind_batch(target_pred)

            target = target.numpy()
            target_pred = target_pred.numpy()

            target = np.clip(target, 0.5, 1.8)
            target = (target - np.min(target)) / (np.max(target) - np.min(target))
            target = (target * 255).astype(np.uint8)
            target = np.squeeze(target)
            target_pred = np.clip(target_pred, 0.5, 1.8)
            target_pred = (target_pred - np.min(target_pred)) / (np.max(target_pred) - np.min(target_pred))
            target_pred = (target_pred * 255).astype(np.uint8)
            target_pred = np.squeeze(target_pred)
            assert target.ndim==2, target_pred.ndim==2
            
            target_vis = cv2.applyColorMap(target, cv2.COLORMAP_TURBO)
            target_pred_vis = cv2.applyColorMap(target_pred, cv2.COLORMAP_TURBO)

            target_vis = np.transpose((target_vis[:, :, :3]), (2, 0, 1)) / 255
            target_vis = torch.from_numpy(target_vis.astype(np.float32))
            target_pred_vis = np.transpose((target_pred_vis[:, :, :3]), (2, 0, 1)) / 255
            target_pred_vis = torch.from_numpy(target_pred_vis.astype(np.float32))

            writer.add_image("Train_gt_depth", target_vis, epoch)
            writer.add_image("train_predicted_depth", target_pred_vis, epoch)

        elif config['output_mod'] == "mask":
            B, H, W = target.squeeze().shape
            target = unbind_batch(target)
            target_pred = torch.clip(target_pred, 0, 1)
            target_pred = unbind_batch((target_pred))

            target = target.squeeze()
            target_pred = target_pred.squeeze()
            target = target.reshape(1, H, W * B)
            target_pred = target_pred.reshape(1, H, W * B)

            writer.add_image("Train_gt_mask", target, epoch)
            writer.add_image("Train_pred_mask", target_pred, epoch)
        target_losses.clear()

        torch.save({
        'Enc' : Encoder.state_dict(),
        'Dec' : Decoder.state_dict()
        }, 'ckpts_{}.pt'.format(config['ckpt']))
        del target, rgb, target_pred

        if(epoch % config['val_freq'] == 0):
            target_vlosses = []
            geom_vlosses = []
            for idx, data in enumerate(val_loader):
                with torch.no_grad():
                    rgb = data['rgb'].to(device).squeeze().contiguous()
                    target = data['target'][:, :3]
                    gt_norm_mask = data['target'][:, -1]
                    gt_norm_mask = gt_norm_mask.to(device)
                    target = target.to(device).contiguous()
                B = rgb.shape[0]
                z, ll_feat = Encoder(rgb)
                target_pred = Decoder(z, ll_feat, rgb.size()[2:])
                if "normal" == config['output_mod']:
                    target_loss = get_loss(target_pred, target, gt_norm_mask).squeeze()
                else:
                    target_loss = criterion(target_pred, target)
                # norm = torch.linalg.norm(target_pred, dim=1)
                # ones = torch.ones_like(norm)
                # geom_loss = criterion(norm, ones).squeeze()
                # val_loss = target_loss + geom_loss
                val_loss = target_loss
                target_vlosses.append(val_loss.item())
                # geom_vlosses.append(geom_loss.item())
            
            losses = np.array(target_vlosses)
            writer.add_scalar("Val Target Loss", np.mean(losses), epoch)
            # losses = np.array(geom_vlosses)
            # writer.add_scalar("Val Geom Loss", np.mean(losses), epoch)
            
            if len(target) > config['log_img_cnt']:
                rgb = rgb[: config['log_img_cnt']].cpu()
                target = target[: config['log_img_cnt']].cpu()
                target_pred = target_pred[: config['log_img_cnt']].cpu()
            else:
                rgb = rgb.cpu()
                target = target.cpu()
                target_pred.cpu()
    
            writer.add_image("Val RGB", torch.clamp(torch.cat(torch.unbind(rgb, dim=0), dim=2), 0, 1), epoch)
            if config['output_mod'] == "normal":
                target = unbind_batch(target)
                target += 1
                target /= 2
                target_pred = unbind_batch(target_pred)
                target_pred += 1
                target_pred /= 2
                target_pred = torch.clip(target_pred, 0, 1)
                writer.add_image("Val GT Normal", target, epoch)
                writer.add_image("Val Pred Normal", target_pred, epoch)

            elif config['output_mod'] == "depth":
                target = unbind_batch(target)
                target_pred = torch.clip(target_pred, 0, 1)
                target_pred = unbind_batch(target_pred)

                target = target.numpy()
                target_pred = target_pred.numpy()
                
                target = np.clip(target, 0.5, 1.8)
                target = (target - np.min(target)) / (np.max(target) - np.min(target))
                target = (target * 255).astype(np.uint8)
                target = np.squeeze(target)
                target_pred = np.clip(target_pred, 0.5, 1.8)
                target_pred = (target_pred - np.min(target_pred)) / (np.max(target_pred) - np.min(target_pred))
                target_pred = (target_pred * 255).astype(np.uint8)
                target_pred = np.squeeze(target_pred)
                assert target.ndim==2, target_pred.ndim==2

                target_vis = cv2.applyColorMap(target, cv2.COLORMAP_TURBO)
                target_pred_vis = cv2.applyColorMap(target_pred, cv2.COLORMAP_TURBO)

                target_vis = np.transpose((target_vis[:, :, :3]), (2, 0, 1)) / 255
                target_vis = torch.from_numpy(target_vis.astype(np.float32))
                target_pred_vis = np.transpose((target_pred_vis[:, :, :3]), (2, 0, 1)) / 255
                target_pred_vis = torch.from_numpy(target_pred_vis.astype(np.float32))

                writer.add_image("val_gt_depth", target_vis, epoch)
                writer.add_image("val_predicted_depth", target_pred_vis, epoch)

            elif config['output_mod'] == "mask":
                B, H, W = target.squeeze().shape
                target = unbind_batch(target)
                target_pred = torch.clip(target_pred, 0, 1)
                target_pred = unbind_batch((target_pred))

                target = target.squeeze()
                target_pred = target_pred.squeeze()
                target = target.reshape(1, H, W * B)
                target_pred = target_pred.reshape(1, H, W * B)

                writer.add_image("val_gt_mask", target, epoch)
                writer.add_image("val_pred_mask", target_pred, epoch)
    
                target_losses.clear()


def unbind_batch(img):
    """Unbind tensor of b x c x h x w and make a single image"""
    with torch.no_grad():
        img = torch.squeeze(img)
        img_unbind = torch.unbind(img)
        img = torch.cat(img_unbind, dim=-1)
        
        return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='.')
    args = parser.parse_args()
    config_file = args.config
    with open(config_file) as c:
        config = json.load(c)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device : {}" .format(device))
    writer = SummaryWriter(comment=config['ckpt'])

    Encoder = Encoder(config).to(device)
    Decoder = Decoder(config).to(device)

    optim = torch.optim.Adam(list(Encoder.parameters()) + list(Decoder.parameters()), lr=1e-4)
    criterion = nn.MSELoss()
    if config['output_mod'] == 'mask':
        criterion = nn.BCELoss()

    random_state = random.randint(1, 100)
    trainset = RGB2NormalDataset(config, True, random_state,
                                transform=transforms.RandomApply([
                                    transforms.RandomResizedCrop(config['img_size'],\
                                        scale=(0.7, 1.0))
                                ], p=0.5),
                                colorjitter=transforms.RandomApply([
                                    transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                                    transforms.GaussianBlur(5),
                                    transforms.RandomErasing(scale=(0.02, 0.2))
                                ], p=0.5))
    valset = RGB2NormalDataset(config, False, random_state)
    train_loader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True,\
        num_workers=config['num_workers'], drop_last=True)
    val_loader = DataLoader(valset, batch_size=config['batch_size'], shuffle=True,\
        num_workers=config['num_workers'], drop_last=True)

    for epoch in tqdm(range(config['num_epochs'])):
        iterate(train_loader, val_loader, device, writer, config, Encoder, Decoder)



