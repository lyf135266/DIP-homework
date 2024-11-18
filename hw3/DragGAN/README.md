# Assignment 3 - Automatic Portrait Editor

### 本次作业采用了DrawGAN与dlib, 实现了一个自动肖像编辑器功能.

### Resources:
- [DragGAN](https://vcai.mpi-inf.mpg.de/projects/DragGAN/): [Implementaion 1](https://github.com/XingangPan/DragGAN)
- [Facial Landmarks Detection by dlib](https://github.com/davisking/dlib)

### 配置说明
本次作业以DrawGAN官方源码为基础，并在此基础上添加opencv-python库以及dlib库，dlib不支持最新的numpy库，需要指定numpy.__version__==1.26.4
DrawGAN配置可参照官方源码，在Windows平台安装有诸多bug，已知的有:
需要在本地安装与虚拟环境版本一致的CUDA
读取字体的大小的函数getsize()被弃用，需要更改为getbbox()
需要安装setuptools
