./data/cat_12_data里应当存放猫的十二分类数据集。

1.运行data文件里的generate_txt.py，会在cat_12_data文件夹下生成train.txt和val.txt用于训练和验证。

2.运行generate_documments.py，会在当前目录下生成processed_data，里边含训练验证所需数据。

3.运行train.py进行训练，训练之前请先前往https://download.pytorch.org/models/resnet34-333f7ec4.pth下载预训练权重，
   并改名为resnet34-pre.pth，训练结束后，会在当前目录下生成best_resNet34.pth，用于对测试的预测。

4.运行batch_predict.py，进行预测，预测结束后，会在当前目录下生成test_list.txt文件，即该次预测结果。