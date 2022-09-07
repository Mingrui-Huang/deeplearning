import shutil
import os

out_root = './processed_data'
if not os.path.exists(out_root):
    os.mkdir(out_root)


def classify_data(txt_name, train_or_val=None, labels=None):
    txt_path = os.path.join('./cat_12_data/', txt_name)
    with open(txt_path, 'r') as fh:
        for line in fh:
            line = line.strip('\n')  # 移除字符串首尾的换行符
            line = line.rstrip()  # 删除末尾空
            words = line.split()  # 以空格为分隔符 将字符串分成两部分
            srcfile = './cat_12_data/' + words[0]
            imgs_label = int(words[1])
            print(srcfile)
            if imgs_label == 0:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[0])
            elif imgs_label == 1:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[1])
            elif imgs_label == 2:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[2])
            elif imgs_label == 3:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[3])
            elif imgs_label == 4:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[4])
            elif imgs_label == 5:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[5])
            elif imgs_label == 6:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[6])
            elif imgs_label == 7:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[7])
            elif imgs_label == 8:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[8])
            elif imgs_label == 9:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[9])
            elif imgs_label == 10:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[10])
            elif imgs_label == 11:
                shutil.copy(srcfile, out_root + '/' + train_or_val + '/' + labels[11])
        print("Copy files Successfully!")


if __name__ == '__main__':

    train_or_val = ["train", "val"]

    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
              '11']

    for i in train_or_val:
        path_train_or_val = os.path.join(out_root, i)
        if not os.path.exists(path_train_or_val):
            os.mkdir(os.path.join(path_train_or_val))
            for j in range(len(labels)):
                path_train_or_val_label = os.path.join(path_train_or_val, labels[j])
                if not os.path.exists(path_train_or_val_label):
                    os.mkdir(path_train_or_val_label)
    classify_data(txt_name='train.txt', train_or_val='train', labels=labels)
    classify_data(txt_name='val.txt', train_or_val='val', labels=labels)
