import timm
import torch
import torch.nn as nn

from model.SSA import Sparse_Self_Attention
from decoder.decoder_be import decoder

from transformers import AutoModel


class SPAM(nn.Module):
    def __init__(self):
        super(SPAM, self).__init__()

        self.spatial_attention = SpatialAttention()
        self.pixel_attention = PixelAttention()

    def forward(self, x):
        spatial_att = self.spatial_attention(x)
        pixel_att = self.pixel_attention(x)
        return spatial_att * pixel_att * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class PixelAttention(nn.Module):
    def __init__(self):
        super(PixelAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_pool, max_pool], dim=1)
        pixel_att = self.conv1(combined)
        pixel_att = self.sigmoid(pixel_att)
        return pixel_att


class BSM(nn.Module):
    """
    边界增强模块
    """

    def __init__(self, channel1, channel2):
        super(BSM, self).__init__()
        channels = channel1 + channel2
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channel1, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel1),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(channel1, channel1, 1)
        # 原始网络sparse_size设置为8
        self.ssa8 = Sparse_Self_Attention(channel1, num_heads=8, sparse_size=4)
        self.spam = SPAM()

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.conv3(x)
        x = self.spam(x)
        if x.shape[1] == 80:
            x = self.ssa8(x) * x + x
        y = self.conv1(x)
        return y


class BENet(nn.Module):
    def __init__(self, f_c=80):
        super(BENet, self).__init__()
        BatchNorm = nn.BatchNorm2d

        # MambaVision
        model_path = r"pretrained/mambavision"
        self.backbone_vmamba = AutoModel.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        resnet34_pth_path = 'pretrained/resnet/resnet34_a1_0-46f8f793.pth'
        self.backbone_res = timm.create_model("resnet34", pretrained=True,
                                              pretrained_cfg_overlay=dict(file=resnet34_pth_path), features_only=True)

        self.decoder = decoder(f_c, BatchNorm)

        self.bsm1 = BSM(80, 64)
        self.bsm2 = BSM(160, 128)
        self.bsm3 = BSM(320, 256)
        self.bsm4 = BSM(640, 512)

        self.conv_final = nn.Sequential(
            nn.ConvTranspose2d(in_channels=160, out_channels=80, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=80, out_channels=40, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.Conv2d(in_channels=40, out_channels=1, kernel_size=1),
        )

    def forward(self, hr_img):
        _, p = self.backbone_vmamba(hr_img)

        r = self.backbone_res(hr_img)

        x1, x2, x3, x4 = self.bsm1(p[0], r[1]), self.bsm2(p[1], r[2]), self.bsm3(p[2], r[3]), self.bsm4(p[3], r[4])

        y = self.decoder(x1, x2, x3, x4)
        output = torch.sigmoid(y)
        return output
