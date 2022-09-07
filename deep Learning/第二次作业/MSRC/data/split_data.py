import os
import random
import shutil


def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.bmp')] #


def move_labels(dest_dir, path):
    label_list = get_imlist(dest_dir + f'/images/{path}')
    for names in label_list:
        name = names.split('\\')[-1:][0]
        name = name.split('.')[0]
        label = name + '.bmp'
        if os.path.exists(f'{label_path}/{label}'):
            shutil.copy(f'{label_path}/{label}', dest_dir + f'/labels/{path}')
        else:
            print(f'{label} is not exists')


def main(src_path):
    dest_dir = output_path  # 这个文件夹需要提前建好
    img_list = get_imlist(src_path)
    random.shuffle(img_list)
    length = int(len(img_list) * split_rate)  # 这个可以修改划分比例
    os.makedirs(dest_dir + '/images/train')
    os.makedirs(dest_dir + '/images/test')
    os.makedirs(dest_dir + '/labels/train')
    os.makedirs(dest_dir + '/labels/test')
    for f in img_list[length:]:
        shutil.copy(f, dest_dir + '/images/train')
    for f in img_list[:length]:
        shutil.copy(f, dest_dir + '/images/test')
    # 移动对应的标签到对应位置
    move_labels(dest_dir, 'test')
    move_labels(dest_dir, 'train')
    print(f'finished')


if __name__ == '__main__':
    path_dataset = r'./msrc_objcategimagedatabase_v2'
    img_path = path_dataset + '/Images'
    label_path = path_dataset + '/GroundTruth'

    split_rate = 0.2

    output_path = './split_data'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    main(img_path)

