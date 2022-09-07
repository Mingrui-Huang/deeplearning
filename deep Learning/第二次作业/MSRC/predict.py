import os
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import time

from unet import Unet


def cam_mask(mask, MSRC_COLORMAP, num_classes):
    seg_img = np.zeros((np.shape(mask)[0], np.shape(mask)[1], 3))
    for c in range(num_classes):
        seg_img[:, :, 0] += ((mask[:, :] == c) * (MSRC_COLORMAP[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((mask[:, :] == c) * (MSRC_COLORMAP[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((mask[:, :] == c) * (MSRC_COLORMAP[c][2])).astype('uint8')
    colorized_mask = Image.fromarray(np.uint8(seg_img))
    return colorized_mask


def save_images(mask, output_path, MSRC_COLORMAP, num_classes, name):
    colorized_mask = cam_mask(mask, MSRC_COLORMAP, num_classes)
    if bilinear:
        name = list(name)
        name.insert(-4, '_bil')
        output_fname = ''.join(name)
        colorized_mask.save(os.path.join(output_path, output_fname))
        print('output: %s' % output_fname)
    else:
        colorized_mask.save(os.path.join(output_path, name))
        print('output: %s' % name)


MSRC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                 [0, 0, 128], [0, 128, 128], [128, 128, 128], [192, 0, 0],
                 [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
                 [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
                 [0, 192, 0], [128, 64, 128], [0, 192, 128], [128, 192, 128],
                 [64, 64, 0], [192, 64, 0]]

# 是否使用双线性插值
bilinear = True
# 是否使用自己找到的图片
use_your_picture = False

if bilinear:
    # weights_path = './checkpoints/best_bil.pt'  # 本地训练的模型
    weights_path = './checkpoints/gpu_groups_weights/best_bil.pt'  # 西电高算GPU集群训练的模型
else:
    # weights_path = './checkpoints/best.pt'
    weights_path = './checkpoints/gpu_groups_weights/best.pt'

output_path = r'./pred_image'
if not os.path.exists(output_path):
    os.makedirs(output_path)
assert os.path.exists(weights_path), f"weights {weights_path} not found."

# get devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create model
model = Unet(in_channels=3, num_classes=22, bilinear=bilinear)
model.load_state_dict(torch.load(weights_path))
model.to(device)
model.eval()  # 进入验证模式

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=mean, std=std)])

with torch.no_grad():

    if use_your_picture:
        root = './test_images/your_pictures'
        name = '4_16_s.jpg'
        img_path = os.path.join(root, name)
        original_img = Image.open(img_path).convert('RGB')
        img = transform(original_img).unsqueeze(0)

        T1 = time.perf_counter()
        prediction = model(img.to(device))
        T2 = time.perf_counter()

        prediction = prediction.squeeze(0).cpu().numpy()
        prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
        save_images(prediction, output_path, MSRC_COLORMAP, num_classes=22, name=name)
        print('inference time: %.3fs' % (T2 - T1))

    else:
        max_num = None
        root = './data/msrc_objcategimagedatabase_v2'

        txt_name = os.path.join(root, 'test.txt')
        with open(txt_name, 'r') as f:
            images = f.read().split()
        if max_num is not None:
            images = images[:min(max_num, len(images))]

        for i in range(len(images)):
            fname = images[i]
            name = (fname + '.bmp')
            print(name)
            original_img = Image.open('%s/Images/%s' % (root, name)).convert("RGB")
            original_img.save('./test_images/%s' % name)
            img = transform(original_img).unsqueeze(0)

            T1 = time.perf_counter()
            prediction = model(img.to(device))
            T2 = time.perf_counter()
            prediction = prediction.squeeze(0).cpu().numpy()
            prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
            save_images(prediction, output_path, MSRC_COLORMAP, num_classes=22, name=name)
            print('inference time: %.3fs' % (T2 - T1))

