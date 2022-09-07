import os
import json
import shutil
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from model import resnet34


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    # 指向需要遍历预测的图像文件夹
    imgs_root = "./data/cat_12_data/cat_12_test/"
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    # 读取指定文件夹下所有jpg图像路径
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".jpg")]
    # 图片保存根路径
    img_out_path = "./have_predicted"
    if not os.path.exists(img_out_path):
        os.mkdir(img_out_path)
    # 读取类别和对应的索引
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # 实例化模型
    net = resnet34()
    # 修改全连接层为12分类
    in_channel = net.fc.in_features
    # 修改全连接层
    net.fc = nn.Sequential(
        nn.Linear(in_channel, 100, bias=True),
        nn.ReLU(),
        nn.Linear(100, 12, bias=True),
    )
    net.to(device)
    # 载入训练好的模型
    weights_path = "./best_resNet34.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    net.load_state_dict(torch.load(weights_path, map_location=device))

    # 预测
    net.eval()
    batch_size = 8  # 每次预测时将多少张图片打包成一个batch
    test_img_label_list = []
    with torch.no_grad():
        for ids in range(0, len(img_path_list) // batch_size):
            img_list = []
            for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                img = Image.open(img_path)
                img = img.convert('RGB')
                img = data_transform(img)
                img_list.append(img)

            # 批量化图片
            # 将img_list列表中的所有图像打包成一个batch
            batch_img = torch.stack(img_list, dim=0)
            # 类别预测
            output = net(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)


            for idx, (prob, c1ass) in enumerate(zip(probs, classes)):
                print("image: {}  class: {}  prob: {:.3}".format(img_path_list[ids * batch_size + idx],
                                                                 class_indict[str(c1ass.numpy())],
                                                                 prob.numpy()))

                test_img_label_list.append(img_path_list[ids * batch_size + idx][19:] + ' ' + class_indict[str(c1ass.numpy())])

                # 把图片放入对应的类别
                img_save_path = os.path.join(img_out_path, class_indict[str(c1ass.numpy())])
                if not os.path.exists(img_save_path):
                    os.mkdir(img_save_path)
                shutil.copy(img_path_list[ids * batch_size + idx], img_save_path)

    # 把测试图片和标签写入txt
    with open('test_list.txt', 'w') as f:
        f.write('\n'.join(test_img_label_list))

if __name__ == '__main__':
    main()
