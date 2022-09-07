import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import resnet34

batch_size = 16
epochs = 3
learning_rate = 0.0002


def main():
    # 检查cuda是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # transform方法
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 获取 train文件夹和val文件夹所在的根路径
    data_root = os.path.abspath(os.path.join(os.getcwd()))  # get data root path
    image_path = os.path.join(data_root, "data", "processed_data")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    # 制作DataLoader
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=4)

    # 查看训练集和测试集数量
    train_num = len(train_dataset)
    val_num = len(validate_dataset)
    print("number of train images：{}\n number of val images:{}.".format(train_num, val_num))

    # 返回每个类别对应的数值映射，键值逆置后，转成json并写入json文件
    class_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in class_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # 选择模型，加载预训练权重
    net = resnet34()
    # 预训练权重下载地址：https://download.pytorch.org/models/resnet34-333f7ec4.pth
    #                  https://download.pytorch.org/models/resnet50-19c8e357.pth
    #                  为了方便，预训练权重改名为resnet34-pre.pth
    model_weight_path = "./resnet34-pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    # 修改全连接层为12分类
    in_channel = net.fc.in_features
    # 修改全连接层
    net.fc = nn.Sequential(
        nn.Linear(in_channel, 100, bias=True),
        nn.ReLU(),
        nn.Linear(100, 12, bias=True),
    )
    net.to(device)
    # summary(net, input_size=(16, 3, 224, 224))
    # 过滤冻结的参数
    for param in net.parameters():
        param.requires_grad = False
    for layer in [net.layer4.parameters(), net.fc.parameters()]:
        for param in layer:
            param.requires_grad = True
    params_non_frozen = filter(lambda p: p.requires_grad, net.parameters())

    # 选择优化器
    optimizer = optim.Adam(params_non_frozen, lr=learning_rate)
    # 选择损失函数
    loss_function = nn.CrossEntropyLoss()

    # 训练与验证部分
    epochs_list = []
    val_acc_list = []
    train_loss_list = []
    for epoch in range(epochs):
        # 训练模式
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        # 验证模式
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # 返回指定维度的最大值和该值所在的索引
                predict_y = torch.max(outputs, dim=1)[1]
                # 计算正确的个数
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        print('[epoch %d/%d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, epochs, running_loss / len(train_loader), val_accurate))

        epochs_list.append(epoch + 1)
        train_loss_list.append(running_loss / len(train_loader))
        val_acc_list.append(val_accurate)

        # 如果当前准确率为历史最高准确率，保存模型
        save_path = './best_resNet34.pth'
        if val_accurate == max(val_acc_list):
            torch.save(net.state_dict(), save_path)
    print('Finished Training')

    # 作图部分
    plt.figure(figsize=(32, 16), dpi=160)
    plt.subplot(121)
    plt.plot(epochs_list, val_acc_list, marker='o', markersize=3)
    for a, b in zip(epochs_list, val_acc_list):
        plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=10)
    plt.title('Accuracy-Epochs')
    plt.xlabel('epoch')
    plt.ylabel('accuracy(%)')
    plt.subplot(122)
    plt.plot(epochs_list, train_loss_list, marker='o', markersize=3)
    for a, b in zip(epochs_list, train_loss_list):
        plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=10)
    plt.title('Loss-Epochs')
    plt.xlabel('epoch')
    plt.ylabel('loss(%)')
    plt.savefig(r'./training_result.png')
    plt.show()


if __name__ == '__main__':
    main()
