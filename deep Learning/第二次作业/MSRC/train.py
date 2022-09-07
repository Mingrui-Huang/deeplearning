import time
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from msrc_train_dataset import load_data_msrc
from unet import Unet


# 计算混淆矩阵
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


# 根据混淆矩阵计算Acc和mIou
def label_accuracy_score(label_trues, label_preds, n_class):
    """
    返回
    像素精确度：acc
    类别平均精确度：acc_cls
    平均交互比：miou
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()                     # 像素准确度 PA=(TP+TN)/(TP+TN+FP+FN)
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)             # 类别精确度 CPA=TP/(TP+FP)
    acc_cls = np.nanmean(acc_cls)                              # 类别平均类精确度 MPA=(CPA1+CPA2+...+CPAn)/n
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.diag(hist) / (
                hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)  # 交互比 IoU=TP/(TP+FP+FN)
        )
    mean_iou = np.nanmean(iou)                                       # 平均交互比 MIoU=(IoU1+IoU2+...+IoUn)/n
    return acc, acc_cls, mean_iou


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 10
batch_size = 8
crop_size = (200, 200)
bilinear = True

train_loader, val_loader = load_data_msrc(batch_size=batch_size, crop_size=crop_size)

in_channels = 3
classes = 22
unet = Unet(in_channels=in_channels, num_classes=classes, bilinear=bilinear)
unet = unet.to(device)

# 设置损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(unet.parameters(), lr=1e-3, momentum=0.7)


# 训练部分
out_dir = "./checkpoints/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# 存储每个epoch的数据，绘图需要
val_acc_list = []
epochs_list = []
acc_epochs = []
acc_cls_epochs = []
mean_iou_epochs = []
for epoch in range(0, epochs):
    time_start = time.perf_counter()
    print('\nEpoch: %d' % (epoch + 1))
    unet.train()
    sum_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        length = len(train_loader)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = unet(images)  # torch.size([batch_size, classes, width, height])
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        predicted = torch.argmax(outputs.data, 1)

        label_pred = predicted.data.cpu().numpy()
        label_true = labels.data.cpu().numpy()
        acc, acc_cls, mean_iou = label_accuracy_score(label_true, label_pred, classes)

        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% | Acc_cls: %.03f%% |Mean_iou: %.3f'
              % (epoch + 1, (batch_idx + 1 + epoch * length), sum_loss / (batch_idx + 1),
                 100. * acc, 100. * acc_cls, mean_iou))
    print('cost time: %3f' % (time.perf_counter() - time_start))

    # 对每个epoch的训练结果进行验证
    print('Waiting Val...')
    mean_iou_sum = 0.0
    mean_acc = 0.0
    mean_acc_cls = 0.0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            unet.eval()
            images, labels = images.to(device), labels.to(device)
            outputs = unet(images)
            predicted = torch.argmax(outputs.data, 1)

            label_pred = predicted.data.cpu().numpy()
            label_true = labels.data.cpu().numpy()
            acc, acc_cls, mean_iou = label_accuracy_score(label_true, label_pred, classes)

            mean_iou_sum += mean_iou
            mean_acc += acc
            mean_acc_cls += acc_cls

            acc_epoch = (100. * mean_acc / len(val_loader))
            acc_cls_epoch = (100. * mean_acc_cls / len(val_loader))
            mean_iou_epoch = mean_iou_sum / len(val_loader)
        print('acc_epoch: %.3f%% | acc_cls_epoch: %.03f%% |mean_iou_epoch: %.3f'
              % (acc_epoch, acc_cls_epoch, mean_iou_epoch))

        epochs_list.append(epoch)
        acc_epochs.append(acc_epoch)
        acc_cls_epochs.append(acc_cls_epoch)
        mean_iou_epochs.append(mean_iou_epoch)

        val_acc_list.append(mean_iou_epoch)
    # 保存最佳模型
    if bilinear:
        if mean_iou_epoch == max(val_acc_list):
            torch.save(unet.state_dict(), out_dir + "best_bil.pt")
            print("save epoch {} unet".format(epoch + 1))
    else:
        if mean_iou_epoch == max(val_acc_list):
            torch.save(unet.state_dict(), out_dir + "best.pt")
            print("save epoch {} unet".format(epoch + 1))

# 绘制结果部分
plt.figure(figsize=(16, 8), dpi=80)
plt.subplot(121)
plt.plot(epochs_list, acc_epochs, marker='o', markersize=3)
plt.plot(epochs_list, acc_cls_epochs, marker='o', markersize=3)
for a, b in zip(epochs_list, acc_epochs):
    plt.text(a, b, '%.1f' % b, ha='center', va='bottom', fontsize=10)
for a, b in zip(epochs_list, acc_cls_epochs):
    plt.text(a, b, '%.1f' % b, ha='center', va='bottom', fontsize=10)
plt.title('Accuracy-Epochs')
plt.xlabel('epoch')
plt.ylabel('accuracy(%)')
plt.legend(['Acc', 'Acc_classes'])

plt.subplot(122)
plt.plot(epochs_list, mean_iou_epochs, marker='o', markersize=3)
for a, b in zip(epochs_list, mean_iou_epochs):
    plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=10)
plt.title('Miou-Epochs')
plt.xlabel('epoch')
plt.ylabel('mean_iou')
plt.legend('Miou')

# 保存绘制的结果
result_path = './result'
if not os.path.exists(result_path):
    os.makedirs(result_path)
if bilinear:
    plt.savefig(r'./result/train_bil.png')
else:
    plt.savefig(r'./result/train.png')
plt.show()





