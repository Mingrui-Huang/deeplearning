1.checkpoints文件夹用于存放运行train.py后产生的权重文件，子文件夹gpu_groups_weights存放GPU集群训练的权重文件
   data文件夹存放数据集文件
   pred_image文件夹存放运行predict.py后产生的图片
   result文件夹存放运行train.py后产生的训练结果图
   test_image文件夹存放一些自己想要测试的图片

2.数据集使用的是MSRC-v2，路径为./data/msrc_objcategimagedatabase_v2
   数据集下载地址：
   https://download.microsoft.com/download/3/3/9/339d8a24-47d7-412f-a1e8-1a415bc48a15/msrc_objcategimagedatabase_v2.zip

3.运行generate_txt.py 后，在./data/msrc_objcategimagedatabase_v2中生成含有训练集，验证集， 测试集名称的的txt文件。

