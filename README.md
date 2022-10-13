# SRCNN

This repository is implementation of the ["Image Super-Resolution Using Deep Convolutional Networks"](https://arxiv.org/abs/1501.00092)(SRCNN)by PyTorch.


## Requirements

- PyTorch 1.0.0
- Numpy 1.15.4
- Pillow 5.4.1
- h5py 2.8.0
- tqdm 4.30.0


## I prepare
- 通过 `python prepare.py` 命令生成训练集和测试集(hdf5文件)
- 需要准备数据集（高清图像）
- 如果直接使用datasets文件夹中的样例hdf5文件，则可以跳过此步
- 相关参数在 prepare.py 文件中修改

## II train
- 通过 `python train.py` 训练模型

- 可以在wandb网站上观察训练过程
  - https://wandb.ai/
  - 可能需要登录账号
- 相关参数在 train.py 文件中修改

## III test
- 通过 `python test.py` 输入单张HR图片进行测试
- 输出由HR下采样得到的LR图像(bicubic),模型预测图象(srcnn),合并的combine图像
- combine图像从左到右依次为LR图像、模型预测图象、HR图像
- 相关参数在 test.py 文件中修改

