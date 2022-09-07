from sklearn.model_selection import train_test_split
import os


imagedir = r'./msrc_objcategimagedatabase_v2/Images'
outdir = r'./msrc_objcategimagedatabase_v2'

images = []
for file in os.listdir(imagedir):
    filename = file.split('.')[0]
    images.append(filename)
# 训练集:测试集:验证集 为 4：2：2
train, test = train_test_split(images, train_size=0.5, random_state=0)
val, test = train_test_split(test, train_size=0.5, random_state=0)

with open(outdir + os.sep + "train.txt", 'w') as f:
    f.write('\n'.join(train))
with open(outdir + os.sep + "val.txt", 'w') as f:
    f.write('\n'.join(val))
with open(outdir + os.sep + "test.txt", 'w') as f:
    f.write('\n'.join(test))
