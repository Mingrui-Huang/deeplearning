from sklearn.model_selection import train_test_split
import os


original_txt = r'./cat_12_data/train_list.txt'
outdir = r'./cat_12_data/'

images = []
with open(original_txt,'r') as file:
    for line in file:
        line = line.strip('\n')  # 移除字符串首尾的换行符
        line = line.rstrip()  # 删除末尾空
        words = line.split()  # 以空格为分隔符 将字符串拆分
        images.append(words[0] + ' '+ str(words[1]))  # imgs中包含有图像路径和标签
# print(images)
print(len(images))
# 训练集:测试集:验证集 为 8:2
train, val = train_test_split(images, train_size=0.8, random_state=0)

# print(train)
print('train:%d' % len(train))
print('val:%d' % len(val))
with open(outdir + os.sep + "train.txt", 'w') as f:
    f.write('\n'.join(train))
with open(outdir + os.sep + "val.txt", 'w') as f:
    f.write('\n'.join(val))

print('txt have been generated')

