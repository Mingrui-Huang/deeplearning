import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):  # mid_channels是第一个卷积层的输出
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),  # 保持特征层大小
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):  # encoder，下采样和两个卷积层
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):  # decoder，上采样，拼接，两个卷积层
    def __init__(self, in_channels, out_channels, bilinear=True):  # 默认双线性插值
        super(Up, self).__init__()
        if bilinear:                                                                              # 双线性插值
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:                                                                                     # 转置卷积
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)  # 转置卷积后，channels减半
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]  # 高度差
        diff_x = x2.size()[3] - x1.size()[3]  # 宽度差

        # padding， 防止输入的图片高宽不是16整数倍造成上采样后无法与之前下采样的图片特征层
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])  # [左，右，上，下]

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):  # 最后一个是1*1的卷积层
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class Unet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 22,
                 bilinear: bool = True,
                 base_c: int = 64):  # base_channel = 64
        super(Unet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)        # 若采用双线性插值，通道数不翻倍，保证能够拼接。
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)  # 这里的in_channel是拼接后的channel,双线性插值out_channel除于4
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.out_conv(x)

        return out

if __name__ == '__main__':
    net = Unet(3, 22)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    print(net)
    print('cuda is available:', torch.cuda.is_available())
