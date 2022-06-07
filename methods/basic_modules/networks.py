
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from . import mod_resnet
from .mod_resnet import model_dirs
from .attentions import CBAM


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class FeatureFusionBlock(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()

        self.block1 = ResBlock(indim, outdim)
        self.attention = CBAM(outdim)
        self.block2 = ResBlock(outdim, outdim)

    def forward(self, x, f16):
        x = torch.cat([x, f16], 1)
        x = self.block1(x)
        r = self.attention(x)
        x = self.block2(x + r)
        #x = self.block2(x)

        return x


# Single object version, used only in static image pretraining
# This will be loaded and modified into the multiple objects version later (in stage 1/2/3)
# See model.py (load_network) for the modification procedure
class ValueEncoderSO(nn.Module):
    def __init__(self, in_dim=1024):
        super().__init__()

        resnet = mod_resnet.resnet18(pretrained=True, extra_chan=1)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 1/4, 64
        self.layer2 = resnet.layer2  # 1/8, 128
        self.layer3 = resnet.layer3  # 1/16, 256

        self.fuser = FeatureFusionBlock(in_dim + 256, 512)

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image, key_f16, mask):
        # key_f16 is the feature from the key encoder
        image = (image - self.mean) / self.std
        f = torch.cat([image, mask], 1)

        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)  # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.layer1(x)  # 1/4, 64
        x = self.layer2(x)  # 1/8, 128
        x = self.layer3(x)  # 1/16, 256

        x = self.fuser(x, key_f16)

        return x


# Multiple objects version, used in other times
class ValueEncoder(nn.Module):
    def __init__(self, in_dim=1024):
        super().__init__()

        resnet = mod_resnet.resnet18(pretrained=True, extra_chan=2)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 1/4, 64
        self.layer2 = resnet.layer2  # 1/8, 128
        self.layer3 = resnet.layer3  # 1/16, 256

        self.fuser = FeatureFusionBlock(in_dim + 256, 512)

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image, key_f16, mask, other_masks):
        # key_f16 is the feature from the key encoder
        image = (image - self.mean) / self.std

        f = torch.cat([image, mask, other_masks], 1)

        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)  # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.layer1(x)  # 1/4, 64
        x = self.layer2(x)  # 1/8, 128
        x = self.layer3(x)  # 1/16, 256

        x = self.fuser(x, key_f16)

        return x


class KeyEncoder(nn.Module):
    def __init__(self, backbone_name='resnet50'):
        super().__init__()
        self.num_features = [1024, 512, 256]   # layer 4, layer 3, layer 2; for resnet > 50
        if backbone_name == 'resnet50':
            self.num_features = [1024, 512, 256]
            # resnet pretrain
            resnet = models.resnet50(pretrained=False)
            resnet.load_state_dict(torch.load(model_dirs['resnet50']))
        elif backbone_name == 'resnet18':
            self.num_features = [256, 128, 64]
            resnet = models.resnet18(pretrained=False)
            resnet.load_state_dict(torch.load(model_dirs['resnet18']))
        else:
            raise KeyError('The backbone {} is not supported yet.'.format(backbone_name))

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.layer2 = resnet.layer2  # 1/8, 512
        self.layer3 = resnet.layer3  # 1/16, 1024 or 256 in resnet 18

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, f):
        f = (f - self.mean) / self.std
        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)  # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        f4 = self.res2(x)  # 1/4, 256
        f8 = self.layer2(f4)  # 1/8, 512
        f16 = self.layer3(f8)  # 1/16, 1024

        return f16, f8, f4


class KeyProjection(nn.Module):
    def __init__(self, indim, keydim):
        super().__init__()
        self.key_proj = nn.Conv2d(indim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)

    def forward(self, x):
        return self.key_proj(x)


# Decoder
class UpsampleBlock(nn.Module):
    def __init__(self, skip_c, up_c, out_c):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1)
        self.out_conv = ResBlock(up_c, out_c)

    def forward(self, skip_f, up_f):
        x = self.skip_conv(skip_f)
        x = x + F.interpolate(up_f, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = self.out_conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, inplanes, mdim=256):
        super().__init__()
        self.compress = ResBlock(inplanes[0], 512)
        self.up_16_8 = UpsampleBlock(inplanes[1], 512, mdim)  # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(inplanes[2], 256, mdim)  # 1/8 -> 1/4

        self.pred = nn.Conv2d(mdim, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, f16, f8, f4, osize):
        x = self.compress(f16)
        x = self.up_16_8(f8, x)
        x = self.up_8_4(f4, x)

        x = self.pred(F.relu(x))

        x = F.interpolate(x, size=osize, mode='bilinear', align_corners=False)
        return x
