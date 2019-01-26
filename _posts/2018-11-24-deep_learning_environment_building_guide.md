---
title: '深度学习环境搭建指南'
layout: post
tags:
  - deep learning
category: Deep Learning
---

## 检查 GPU 是否支持

如果你想搭建深度学习环境，那么首先得有一块 NVIDIA GPU。很遗憾，目前 AMD 系列 GPU 对深度学习并不友好。

有了 NVIDIA GPU 之后，首先需要检查其型号是否符合深度学习的最低配置，目前热门的深度学习框架对老旧型号的 NVIDIA GPU 并不支持。其中，判断的原则是 NVIDIA GPU 的计算性能指数（Compute Capability）**大于或等于 3.0**。 你可以访问 [官方性能指数](https://developer.nvidia.com/cuda-gpus) 页面查看。

<!--more-->

![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/dlebg_01.png)

## 更新 NVIDIA GPU 驱动

接下来，你需要更新 NVIDIA GPU 驱动到最新版本。这一步骤非常简单，只需要到 [官方驱动页面](https://www.nvidia.com/Download/index.aspx) 找到对应型号下载安装即可。

![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/dlebg_02.png)

## 安装 CUDA 架构

CUDA（Compute Unified Device Architecture，统一计算架构）是由 NVIDIA 所推出的一种集成技术，深度学习需要 CUDA 并行计算架构的支持。

目前，CUDA 的版本是 `9.x`，这里推荐 `9.0` 版本即可，你可以访问 [官方页面](https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=deblocal) 下载。

![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/dlebg_03.png)

目前，CUDA 支持 Windows，Linux 和 macOS 操作系统。

## 安装 cuDNN 深度神经网络库

cuDNN 是 NVIDIA 为深度学习专门研发的神经网络加速库，并支持当前主流的 TensorFlow、PyTorch 等深度行学习框架。根据官方介绍，cuDNN 能更好地调度 GPU 资源，使得神经网络的训练过程更加高效。

安装 cuDNN 首先需要访问 [NVIDIA Developer 网站](https://developer.nvidia.com/cudnn) 注册账号。该网站国内访问速度慢，可能需要耐心等待或采取其他手段。

![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/dlebg_04.png)

登陆之后，需要根据之前安装的 CUDA 版本选择对应版本的 cuDNN 安装。

![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/dlebg_05.png)

## 安装结果检查

更新驱动 + 安装 CUDA + 安装 cuDNN 等三个步骤完成之后，我们需要检查深度学习框架是否能正常监测到 GPU 并调用。

![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/dlebg_06.png)

这里推荐使用 TensorFlow，首先需要安装 GPU 版本的 TensorFlow。你可以阅读 [官方安装指南](https://www.tensorflow.org/install/)。

安装完成之后，运行 TensorFlow 官方给出的 GPU 检测示例代码即可：

```python
# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
```

如果能在输出中看到 GPU 的字样，即代表安装成功。例如：

```python
Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: Tesla K40c, pci bus
id: 0000:05:00.0
b: /job:localhost/replica:0/task:0/device:GPU:0
a: /job:localhost/replica:0/task:0/device:GPU:0
MatMul: /job:localhost/replica:0/task:0/device:GPU:0
[[ 22.  28.]
 [ 49.  64.]]
```

## 使用云主机搭建

本地安装深度学习环境看起来很简单，但实际上因为系统环境等原因很容易碰到各种「坑」。很多时候，也非常推荐在云主机上完成深度学习训练。

当你选择在云主机上进行深度学习时，往往不需要自行配置环境，而是直接启动已配置好的镜像。无论是国内的 [阿里云](https://market.aliyun.com/)，还是国外的 [AWS](https://aws.amazon.com/marketplace)，都会提供最新的深度学习镜像。

例如，阿里云提供的 [Ubuntu16.04（预装 NVIDIA GPU 驱动和 CUDA9.0）](https://market.aliyun.com/products/57742013/cmjj022697.html?spm=5176.730006-53366009-57742013-cmjj021670/A.recommend.9.xDBKmW) 镜像，已经为你配置好了驱动、CUDA、cuDNN。

![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/dlebg_07.png)

你只需要在启动实例时加载即可，这样就省去了自行配置环境的麻烦。

除此之外，AWS 提供的 [Amazon SageMaker 服务](https://aws.amazon.com/sagemaker) 可以让你一键启动 Jupyter Notebook，并在后端挂载相应的 GPU 实例，可以说再方便不过了。

![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/dlebg_08.png)

## 在线 Notebook 环境推荐

除了自行搭建环境，在此推荐 2 个免费的在线 Jupyter Notebook 环境。

### Microsoft Azure Notebooks

[Microsoft Azure Notebooks](https://notebooks.azure.com/) 是微软推出的免费 Jupyter Notebook 环境，非常方便。不过，Microsoft Azure Notebooks 仅提供 CPU 环境，无法完成深度学习模型训练。但是，平常的数据分析任务处理起来游刃有余了。

![image](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/dlebg_09.png)

### Google Colab

[Google Colab](https://colab.research.google.com/) 是 Google 推出的线上 Notebook 环境。相对于 Microsoft Azure Notebooks 而言，Google Colab 的最大问题存在一些访问上的阻碍。不过，Google Colab 的最大优势在于提供了 Nvidia M40 系列 GPU，并免费开放使用。

使用时，只需要在 **代码执行程序 → 更改运行类型** 中选择 GPU，即可开启免费 GPU 环境。

![image](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/dlebg_10.png)

Google Colab 提供的免费 GPU 并不是无限制使用，如果你请求频次过高或者连续运行时间太长，都有可能被强制中断。