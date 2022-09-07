import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """pytorch tensor通道顺序 [batch, channel, height, width]"""
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1, 2)    # 卷积层1, padding = 2 保证输入输出同尺寸
        '''Conv2d(in_channels, out_channels, kernel_size, stride=, padding)'''
        self.pool1 = nn.MaxPool2d(2, 2)          # 池化层1
        '''MaxpoolNd('kernel_size', 'stride', ...)'''
        self.conv2 = nn.Conv2d(6, 16, 5)         # 卷积层2
        self.pool2 = nn.MaxPool2d(2, 2)          # 池化层2
        self.fc1 = nn.Linear(16*5*5, 120)        # 全连接层1
        '''__constants__ = ['in_features', 'out_features']'''
        self.fc2 = nn.Linear(120, 84)            # 全连接层2
        self.fc3 = nn.Linear(84, 10)             # 全连接层3

    def forward(self, x):            # input x
        x = F.relu(self.conv1(x))    # input(1*28*28) output(6*28*28)
        x = self.pool1(x)            # output(6*14*14)
        x = F.relu(self.conv2(x))    # output(16*10*10)
        x = self.pool2(x)            # output(16*5*5)
        x = x.view(-1, 16*5*5)       # output(16*5*5)
        '''view()把数据展平为一维数据view(-1, ***)表示动态调整该维度上的元素个数，保证元素总数不变'''
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
        return x

if __name__ =='__main__':
    input1 = torch.rand([1, 1, 28, 28])
    model = LeNet()
    print(model)
    output = model(input1)
