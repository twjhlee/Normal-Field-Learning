import torch
from torch import nn, optim
from torch.nn import functional as F
import torchvision
import pdb

def get_Enc_loss(recon, image, t, t_hat, r, r_hat, VGG):
    recon_loss = F.mse_loss(recon.view(-1), image.view(-1))
    perc_loss = VGG(recon, image)
    geo_loss = F.mse_loss(t, t_hat) + F.mse_loss(r, r_hat)
    return recon_loss + perc_loss + geo_loss

def get_Dec_loss(recon, image ,VGG):
    recon_loss = F.mse_loss(recon.view(-1), image.view(-1))
    perc_loss = VGG(recon, image)
    return recon_loss + perc_loss

def get_Gue_loss(t, t_hat, r, r_hat):
    return F.mse_loss(t, t_hat) + F.mse_loss(r, r_hat)

def get_recon_loss(recon, image):
    return F.mse_loss(recon.view(-1), image.view(-1))

def get_geo_loss(t, t_hat, r, r_hat):
    return F.mse_loss(t, t_hat) + F.mse_loss(r, r_hat)

def get_edge_loss(input, recon):
    return F.binary_cross_entropy(recon.view(-1), input.view(-1))

def get_text_inv_loss(z_t, z_o, z_half):
    z_1 = z_t - z_half
    z_2 = z_half - z_o
    return F.mse_loss(z_1.view(-1), z_2.view(-1))


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss

def FocalLoss(logit, target, weight, gamma=2, alpha=0.5, ignore_index=255, size_average=True, batch_average=True):
    n, h, w = logit.shape
    target = target.squeeze(1)
    criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index,
                                   reduction='sum')
    logpt = -criterion(logit, target.long())
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt

    if batch_average:
        loss /= n

    return loss