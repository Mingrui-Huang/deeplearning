import torchvision
import torch.nn as nn
import torch.optim
import torchvision.transforms as transforms
import time

from lenet import LeNet

epochs = 10
batch_size = 32
learning_rate = 0.001

transform = transforms.Compose(
    [transforms.ToTensor(),  # (C x H x W) to [0.0, 1.0]
     ])

train_set = torchvision.datasets.FashionMNIST(root='../../data/',
                                              train=True,
                                              download=False,
                                              transform=transform)
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=batch_size,
                                           shuffle=True)  # 每批batch_size张，打乱顺序
test_set = torchvision.datasets.FashionMNIST(root='../../data/',
                                             train=False,
                                             download=False,
                                             transform=transform)
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=10000,
                                          shuffle=False)

train_data_iter = iter(train_loader)
train_image, train_label = train_data_iter.next()
test_data_iter = iter(test_loader)  # 迭代器
test_image, test_label = test_data_iter.next()

net = LeNet()
loss_function = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # Adam优化器，把LeNet里能训练的参数进行训练

print("Start training!")
best_acc = 0.0
save_path = './LeNet.pth'
for epoch in range(epochs):  # 训练集迭代次数

    running_loss = 0.0  # 累加损失
    time_start = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):
        inputs, labels = data
        optimizer.zero_grad()  # 清除历史梯度，防止累加
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
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
            outputs = net(test_images)
            predict_y = torch.max(outputs, dim=1)[1]  # 输出10个节点中，找最大值作为输出 [1]即索引最大值是啥
            acc += (predict_y == test_labels).sum().item()
        acc_test = acc / len(test_labels)

        if acc_test > best_acc:
            best_acc = acc_test
            torch.save(net.state_dict(), save_path)

        print('[epoch %d/%d] test_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, epochs, running_loss / step, acc / test_labels.size(0)))
        running_loss = 0.0
    print('Finished this epoch test!')
print('History best accuracy: %.3f' % best_acc)

