import torch.nn as nn
import torch


# ResNet-18/34的残差模块
class BasicBlock(nn.Module):
    expansion = 1  # 第二层的卷积核个数为第一层的1背

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        # ---------------------------------------------------------------------------第一层卷积层
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        # ---------------------------------------------------------------------------第二层卷积层
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        # 如果下采样不为空，使用捷径分支为虚线的残差结构
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 主分支输出加上捷径分支输出
        out += identity
        out = self.relu(out)

        return out


# ResNet-50/101/152残差模块
class Bottleneck(nn.Module):
    """
    原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4  # 第三层卷积层的卷积核个数为第一、第二层的4背

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups
        # ----------------------------------------------------------------------------------第一层
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # ----------------------------------------------------------------------------------第二层
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,      # 如果是实线残差结构，stride=1
                               kernel_size=3, stride=stride, bias=False, padding=1)       # 虚线残差结构，stride=2
        self.bn2 = nn.BatchNorm2d(width)
        # ----------------------------------------------------------------------------------第三层
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,  # 第三层卷积核个数为前面卷积层的4倍
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        # 如果下采样不为空，使用捷径分支为虚线的残差结构
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # # 主分支输出加上捷径分支输出
        out += identity
        out = self.relu(out)

        return out

# 定义ResNet框架
class ResNet(nn.Module):
    """
    block:对应的残差结构，ResNet-18/34为BasicBlock，ResNet-50/101/152为Bottleneck
    block_num：表示残差结构的个数，如ResNet的block_num为[3, 4, 6, 3],即conv2_x、conv3_x、conv4_x、conv_5_x分别有3、4、6、3个blocks
    num_classes:输出类别数

    """
    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # conv2_x
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        # conv3_x
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        # conv4_x
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        # conv5_x
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        # 自适应下采样到高宽为1x1
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        # 卷继层初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    # block:BasicBlock或者Bottleneck  channel：第一层卷积核个数  block_num为该conv_x残差结构个数  stride除了conv2_x默认为1，其他conv_x为2
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        # 判断是否是50层以上的ResNet，50层以上的conv2_x第一个残差结构是虚线捷径分支，但只需要调整channel，不调整高和宽，故stride=1
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        # 添加conv2_x, ResNet-18/34的第一个残差结构downsample=None，ResNet-50/101/152的第一个残差结构downsample经过if判断后得到的
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion
        # 添加该conv_x剩余的残差结构
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
