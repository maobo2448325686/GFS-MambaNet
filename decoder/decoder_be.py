import torch
import torch.nn as nn
from model import attention


class SCAM(nn.Module):
    def __init__(self, in_d, out_d):
        super(SCAM, self).__init__()
        self.in_d = in_d * 2
        self.out_d = in_d // 2
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.in_d, out_channels=self.out_d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(),
        )
        self.spa = attention.SpatialAttention()
        self.cha = attention.ChannelAttention(in_d)

    def forward(self, input):
        x1 = self.spa(input) * input
        x2 = self.cha(input)
        output = torch.cat((x1, x2), 1)
        return self.conv1(output)


class Decoder(nn.Module):
    def __init__(self, fc, BatchNorm):
        super(Decoder, self).__init__()
        self.fc = fc
        self.scam1 = SCAM(80, 40)
        self.scam2 = SCAM(160, 80)
        self.scam3 = SCAM(320, 160)
        self.scam4 = SCAM(640, 320)
        self.last_conv = nn.Sequential(nn.ConvTranspose2d(in_channels=40, out_channels=20, kernel_size=4, stride=2, padding=1),
                                       BatchNorm(20),
                                       nn.ReLU(),
                                       # nn.Dropout(0.5),
                                       nn.Conv2d(in_channels=20, out_channels=1, kernel_size=1)
                                       )

        self._init_weight()

    def forward(self, feat1, feat2, feat3, feat4):

        x4 = self.scam4(feat4)
        x3 = self.scam3(feat3+x4)
        x2 = self.scam2(feat2+x3)
        x1 = self.scam1(feat1+x2)
        x = self.last_conv(x1)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def decoder(fc, BatchNorm):
    return Decoder(fc, BatchNorm)
