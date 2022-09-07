import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import sys
sys.path.append("..")
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")  # 忽略警告

MSRC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                 [0, 0, 128], [0, 128, 128], [128, 128, 128], [192, 0, 0],
                 [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
                 [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
                 [0, 192, 0], [128, 64, 128], [0, 192, 128], [128, 192, 128],
                 [64, 64, 0], [192, 64, 0]]

MSRC_CLASS = ['void', 'building', 'grass', 'tree',
              'cow', 'sheep', 'sky', 'aeroplane',
              'water', 'face', 'car', 'bycycle',
              'flower', 'sign', 'bird', 'book',
              'chair', 'road', 'cat', 'dog',
              'body', 'boat']


def read_msrc_images(root="./data/msrc_objcategimagedatabase_v2", is_train=True, max_num=None):
    txt_fname = os.path.join(root, 'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()  # 拆分成一个个名字组成list
    if max_num is not None:  # max_num限制读取的图片数量
        images = images[:min(max_num, len(images))]
    features, labels = [None] * len(images), [None] * len(images)
    for i, fname in tqdm(enumerate(images)):
        # 读入数据并且转为RGB的 PIL image
        features[i] = Image.open('%s/Images/%s.bmp' % (root, fname)).convert("RGB")
        labels[i] = Image.open('%s/GroundTruth/%s_GT.bmp' % (root, fname)).convert("RGB")
    return features, labels  # PIL image 0-255


def msrc_colormap2label():
    """构建从RGB到msrc类别索引的映射"""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(MSRC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label


# 构造标签矩阵
def msrc_label_indices(colormap, colormap2label):
    colormap = np.array(colormap.convert("RGB")).astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])
    return colormap2label[idx]  # colormap 映射 到colormaplabel中计算的下标


def msrc_rand_crop(feature, label, height, width):
    i, j, h, w = torchvision.transforms.RandomCrop.get_params(feature, output_size=(height, width))
    feature = torchvision.transforms.functional.crop(feature, i, j, h, w)
    label = torchvision.transforms.functional.crop(label, i, j, h, w)
    return feature, label


class MSRCSegDataset(Dataset):
    def __init__(self, is_train, crop_size, root, max_num=None):
        """
        crop_size: (h, w)
        """
        # 对输入图像的RGB三个通道的值分别做标准化
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)])
        self.crop_size = crop_size  # (h, w)
        features, labels = read_msrc_images(root=root, is_train=is_train, max_num=max_num)
        # 由于数据集中有些图像的尺寸可能小于随机裁剪所指定的输出尺寸，这些样本需要通过自定义的filter函数所移除
        self.features = self.filter(features)  # PIL image
        self.labels = self.filter(labels)  # PIL image
        self.colormap2label = msrc_colormap2label()
        if is_train:
            print('训练集加载了 ' + str(len(self.features)) + ' 张有效图片')
        else:
            print('验证集加载了 ' + str(len(self.features)) + ' 张有效图片')

    def filter(self, imgs):
        return [img for img in imgs if (
                img.size[1] >= self.crop_size[0] and img.size[0] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = msrc_rand_crop(self.features[idx], self.labels[idx], *self.crop_size)
        # float32 tensor           uint8 tensor (b,h,w)
        return self.transform(feature), msrc_label_indices(label, self.colormap2label)

    def __len__(self):
        return len(self.features)


def load_data_msrc(batch_size, crop_size):
    """加载msrc语义分割数据集"""
    root = r"./data/msrc_objcategimagedatabase_v2"
    train_loader = DataLoader(
        MSRCSegDataset(is_train=True, crop_size=crop_size, root=root), batch_size,
        shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(
        MSRCSegDataset(is_train=False, crop_size=crop_size, root=root), batch_size,
        drop_last=True, num_workers=0)

    return train_loader, val_loader


def show_images(imgs, num_rows, num_cols, scale=2):
    # a_img = np.asarray(imgs)
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    plt.show()
    return axes

if __name__ == "__main__":

    root = r"./data/msrc_objcategimagedatabase_v2"
    train_features, train_labels = read_msrc_images(root, max_num=10)
    n = 5  # 展示几张图像
    imgs = train_features[0:n] + train_labels[0:n]  # PIL image
    show_images(imgs, 2, n)

    # colormap2label = msrc_colormap2label()
    # y = msrc_label_indices(train_labels[0], colormap2label)
    # torch.set_printoptions(profile="full")
    # print((y[:, :]), train_labels[1])

    imgs = []
    for _ in range(n):
        imgs += msrc_rand_crop(train_features[0], train_labels[0], 160, 150)
    show_images(imgs[::2] + imgs[1::2], 2, n)

    batch_size = 32
    crop_size = (200, 200)  # 指定随机裁剪的输出图像的形状为(200,200)
    train_loader, val_loader = load_data_msrc(batch_size, crop_size)

    train_data_iter = iter(train_loader)
    train_images, train_labels = train_data_iter.next()
    print(train_images.shape)
