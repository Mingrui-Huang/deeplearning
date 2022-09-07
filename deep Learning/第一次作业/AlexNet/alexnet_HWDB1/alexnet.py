import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)      # 卷积层1
        self.pool1 = nn.MaxPool2d(2, 2)             # 池化层1
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)     # 卷积层2
        self.pool2 = nn.MaxPool2d(2, 2)             # 池化层2
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)    # 卷积层3
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)   # 卷积层4
        self.conv5 = nn.Conv2d(256, 256, 3, 1, 1)   # 卷积层5
        self.pool3 = nn.MaxPool2d(3, 2)             # 池化层3
        self.fc1 = nn.Linear(256*3*3, 1024)         # 全连接层1
        self.drop1 = nn.Dropout(0.5)                # 失活层1,有0.5的概率失活
        self.fc2 = nn.Linear(1024, 1024)             # 全连接层2
        self.drop2 = nn.Dropout(0.5)                # 失活层2
        self.fc3 = nn.Linear(1024, 10)               # 全连接层3

    def forward(self, x):                           # input(x)
        x = self.conv1(x)                           # input(1*28*28)  output(32*28*28)
        x = F.relu(self.pool1(x))                   # output(32*14*14)
        x = self.conv2(x)                           # output(64*14*14)
        x = F.relu(self.pool2(x))                   # output(64*7*7)
        x = self.conv3(x)                           # output(128*7*7)
        x = self.conv4(x)                           # output(256*7*7)
        x = self.conv5(x)                           # output(256*7*7)
        x = self.pool3(x)                           # output(256*3*3)
        x = x.view(-1, 256*3*3)                     # 展平成一维（2304）
        x = F.relu(self.fc1(x))                     # output(1024)
        x = self.drop1(x)
        x = F.relu(self.fc2(x))                     # output(512)
        x = self.drop2(x)
        x = self.fc3(x)                             # output(10)
        return x
if __name__ =='__main__':
    input1 = torch.rand([1, 1, 28, 28])
    model = AlexNet()
    print(model)
    output = model(input1)
