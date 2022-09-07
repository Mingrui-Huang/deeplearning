from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2


# 自定义图片图片读取方式
def MyLoader(path):
    #img = Image.open(path).convert('RGB')
    img = cv2.imread(path)  # 读入图片
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图片转为灰度图
    img = cv2.equalizeHist(img)  # 直方图均衡化，增加对比度
    img = cv2.resize(img, (28, 28))  # 统一大小
    return img


class MyDataset(Dataset):
    # 构造函数设置默认参数
    def __init__(self, txt, transform=None, target_transform=None, loader= MyLoader):
        with open(txt, 'r') as fh:
            imgs = []
            for line in fh:
                line = line.strip('\n')  # 移除字符串首尾的换行符
                line = line.rstrip()  # 删除末尾空
                words = line.split()  # 以空格为分隔符 将字符串拆分
                imgs.append((words[0], int(words[1])))  # imgs中包含有图像路径和标签
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        # 调用定义的loader方法
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    transform = transforms.Compose([
        # transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    with open(r'./DATA/HWDB1/train.txt', 'r') as fh:
        imgs = []
        for line in fh:
            line = line.strip('\n')  # 移除字符串首尾的换行符
            line = line.rstrip()  # 删除末尾空
            words = line.split()  # 以空格为分隔符 将字符串拆分
            imgs.append((words[0], int(words[1])))  # imgs中包含有图像路径和标签
    print(imgs)
    print(len(imgs))
    # for i in range(len(imgs)):
    #     fn, label = imgs[i]
    #     img = MyLoader(fn)
    #     print(img.shape)
    #     cv2.imshow('%d %s' % (i, label), img)
    #     cv2.waitKey()
    root = r"./DATA/HWDB1"
    train_data = MyDataset(txt=root + '\\' + 'train.txt', transform=transform)
    test_data = MyDataset(txt=root + '\\' + 'test.txt', transform=transform)

    # train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=32)
    print('加载成功！')

    train_data_iter = iter(train_loader)
    train_image, train_label = train_data_iter.next()
    test_data_iter = iter(test_loader)  # 迭代器
    test_image, test_label = test_data_iter.next()
    print(test_image.shape)
    print(train_image.shape)
