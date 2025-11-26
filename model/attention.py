import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        '''
        Args:
            in_planes: in_planes: 输入特征图的通道数
            ratio: ratio: 压缩比例，默认为8.用于控制通道压缩后的维度
        '''
        super(ChannelAttention, self).__init__()
        # 自适应平均池化层，将输入特征图进行平均池化层，输出大小为1 × 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 自适应最大池化层，将输入特征图进行最大池化，输出大小为1 × 1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 1 × 1 卷积层，用于对输入特征图进行通道压缩。输入通道数为in_planes，输出通道数为in_planes // ratio
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        # ReLU激活函数，增加模型的非线性能力。
        self.relu1 = nn.ReLU()
        # 1×1 卷积层，用于恢复压缩后的通道数。输入通道数为in_planes // ratio，输出通道数为in_planes
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        # Sigmoid激活函数，将输出的特征图缩放到0和1之间。
        self.sigmoid = nn.Sigmoid()

    # 在前向传播方法中，实现了通道注意力的计算过程
    def forward(self, x):
        # 首先，通过将输入特征图分别经过平均池化和最大池化操作，得到两个大小为 1×1 的特征图 avg_out和 max_out
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # 接着，对 avg_out 和 max_out 分别应用通道压缩操作，即先经过 fc1 和 relu1，再经过 fc2，得到压缩后的特征图。
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # 最后，将压缩后的特征图相加，得到最终输出的特征图。并通过Sigmoid函数进行缩放，将特征图中每个通道的值映射到 0 和 1 之间
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=3):
        '''
        Args:
            kernel_size: 卷积核的尺寸，默认值为3.只支持 3 或 7 这两个尺寸。
        '''
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # 卷积层，用于对输入特征图进行卷积操作。输入通道数为2（由后面的torch.cat函数决定），输出通道数为 1。卷积核的尺寸为 kernel_size, padding 参数根据 kernel_size 的取值确定
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        # Sigmoid激活函数，将输出的特征图缩放到 0 和 1 之间
        self.sigmoid = nn.Sigmoid()

    # 在前向传播方法中，实现了空间注意力的计算过程
    def forward(self, x):
        base = x
        # 通过对输入特征图在通道维度上进行均值池化，得到大小为 1×H×W 的特征图 avg_out （H 和 W 为特征图的高度和宽度）
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 通过对输入特征图在通道维度上你进行最大值池化，得到大小为 2×H×W 的特征图 max_out。
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 将 avg_out和 max_out 按通道维度进行拼接，得到大小为 2×H×W 的特征图 x 。
        x = torch.cat([avg_out, max_out], dim=1)
        # 将图征途输入到卷积层conv1中进行卷积操作，得到输出特征图。并通过Sigmoid函数进行缩放，将特征图中的值映射到0和1之间
        x = self.conv1(x)
        x = self.sigmoid(x)
        return x * base
