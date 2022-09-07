import torchvision
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time

from Mydata import MyDataset
from alexnet import AlexNet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 查看cuda是否可用，使用cuda加速
print(device)

epochs = 30
batch_size = 32
learning_rate = 0.001

root = r'./DATA/HWDB1'

transform = transforms.Compose([
    #transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

train_data = MyDataset(txt=root + '\\' + 'train.txt', transform=transform)
test_data = MyDataset(txt=root + '\\' + 'test.txt', transform=transform)

# train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size)

print('加载成功！')

train_data_iter = iter(train_loader)
train_image, train_label = train_data_iter.next()
test_data_iter = iter(test_loader)  # 迭代器
test_image, test_label = test_data_iter.next()
print(test_image.shape)
print(train_image.shape)

'''显示图片'''
classes = ('一', '丁', '七', '万',
           '丈', '三', '上', '下', '不', '与')

def imshow(img):
    img = img / 2 + 0.5    # 反标准化  input = 0.5 * output + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))  # (high, width, channel)
    plt.show()
# print labels
print(' '.join(f'{classes[train_label[j]]:2s}' for j in range(batch_size)))
# show images
imshow(torchvision.utils.make_grid(train_image))

net = AlexNet()
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

print("Start Training!")
best_acc = 0.0
save_path = './AlexNet.pth'
for epoch in range(epochs):

    running_loss = 0.0  # 累加损失
    time_start = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):
        inputs, labels = data
        optimizer.zero_grad()  # 清除历史梯度，防止累加
        outputs = net(inputs.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end=".")
    print()
    print('cost time: %3f' % (time.perf_counter() - time_start))
    print('Finished this epoch training!')

    acc = 0.0
    with torch.no_grad():

        for data_test in test_loader:
            test_images, test_labels = data_test
            outputs = net(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]  # 输出10个节点中，找最大值作为输出 [1]即索引最大值是啥
            acc += (predict_y == test_labels.to(device)).sum().item()
        acc_test = acc / len(test_labels)

        if acc_test > best_acc:
            best_acc = acc_test
            torch.save(net.state_dict(), save_path)

        print('[epoch %d/%d] test_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, epochs, running_loss / step, acc / test_labels.size(0)))
        running_loss = 0.0
    print('Finished this epoch test!')
print('History best accuracy: %.3f' % best_acc)
