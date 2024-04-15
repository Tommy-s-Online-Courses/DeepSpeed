# -*- coding: utf-8 -*-
"""
Training a Classifier
=====================

This is it. You have seen how to define neural networks, compute loss and make
updates to the weights of the network.

Now you might be thinking,

What about data?
----------------

Generally, when you have to deal with image, text, audio or video data,
you can use standard python packages that load data into a numpy array.
Then you can convert this array into a ``torch.*Tensor``.

-  For images, packages such as Pillow, OpenCV are useful
-  For audio, packages such as scipy and librosa
-  For text, either raw Python or Cython based loading, or NLTK and
   SpaCy are useful

Specifically for vision, we have created a package called
``torchvision``, that has data loaders for common datasets such as
Imagenet, CIFAR10, MNIST, etc. and data transformers for images, viz.,
``torchvision.datasets`` and ``torch.utils.data.DataLoader``.

This provides a huge convenience and avoids writing boilerplate code.

For this tutorial, we will use the CIFAR10 dataset.
It has the classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’,
‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. The images in CIFAR-10 are of
size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.

.. figure:: /_static/img/cifar10.png
   :alt: cifar10

   cifar10


Training an image classifier
----------------------------

We will do the following steps in order:

1. Load and normalizing the CIFAR10 training and test datasets using
   ``torchvision``
2. Define a Convolutional Neural Network
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data

1. Loading and normalizing CIFAR10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using ``torchvision``, it’s extremely easy to load CIFAR10.
"""
import torch
import torchvision
import torchvision.transforms as transforms

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].
# .. note::
#     If running on Windows and you get a BrokenPipeError, try setting
#     the num_worker of torch.utils.data.DataLoader() to 0.


transform = transforms.Compose([
    # 将一个 PIL Image 或者 numpy.ndarray（H x W x C）转化为 torch.Tensor（C x H x W），
    # 并且归一化到 [0, 1]。这一步是为了把图像转化为 PyTorch 可以处理的数据类型
    transforms.ToTensor(),
    # Normalize 这个函数是进行标准化处理。它接收两个参数：一个是均值，一个是标准差。
    # 这里的 (0.5, 0.5, 0.5) 表示 RGB 三个通道的均值和标准差。
    # 对每个通道进行如下操作： image = (image - mean) / std,
    # 假设原来的像素值是 [0, 1]，那么这个操作之后的像素值就会变成 [-1, 1],
    # 这个操作可以使得模型训练过程中的数值更稳定。
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# CIFAR-10 是一个常用的图像分类数据集，包含 60000 张 32x32 的彩色图片，有 10 个类别。

trainset = torchvision.datasets.CIFAR10(root='./data', # 指定了数据集的下载路径
                                        train=True, # 表示加载的是训练集
                                        download=True, # 表示如果数据集没有在指定路径下找到，那么就下载数据集。如果数据集已经存在，那么这个参数没有作用。
                                        transform=transform # 指定了一个数据预处理的函数
                                        )
# 创建一个数据加载器，这个加载器可以在训练模型时批量加载数据。
trainloader = torch.utils.data.DataLoader(trainset, # 是你要加载的数据集
                                          batch_size=4, # 指定每个 batch 的大小，也就是每次模型训练时输入的数据量。
                                          shuffle=True, # 指在每个 epoch 开始时，要不要把数据打乱。这在训练模型时是一种常用的方式，可以增强模型的泛化能力。
                                          num_workers=12 # 指定使用多少个子进程来加载数据
                                          )

testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=4,
                                         shuffle=False,
                                         num_workers=12)
# 10个类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')

########################################################################
# Let us show some of the training images, for fun.

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    # 显示一个图像,这个图像是一个PyTorch的Tensor，具有3个维度，分别代表颜色通道、高度和宽度 (C, H, W)
    img = img / 2 + 0.5  # 对图像数据进行反标准化, 标准化后的图像的像素值是在[-1, 1]之间, 反标准化就是将这些值再变回原来的[0, 1]区间。
    npimg = img.numpy() # 将 Tensor 转化为 NumPy 数组
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # 使用了np.transpose方法对图像的维度进行了调换，Matplotlib希望的输入是（H, W, C）形式的，所以需要进行转换。
    plt.show()


# get some random training images
# 创建了一个迭代器，这个迭代器可以从trainloader中一次获取一个batch的数据。
dataiter = iter(trainloader)
# 使用next函数从dataiter中获取了一个batch的数据，这个batch包含了4张图像（因为batch_size=4）以及这些图像对应的标签。
images, labels = next(dataiter)

# show images
# 使用了torchvision.utils.make_grid函数，这个函数将多张图像拼接在一起，
# 形成一个网格状的大图像。然后用 imshow 函数将这个大图像显示出来。
imshow(torchvision.utils.make_grid(images))
# 打印出每张图像对应的标签
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

########################################################################
# 2. Define a Convolutional Neural Network
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    '''特征图尺寸的计算公式为：[(原图片尺寸 — 卷积核尺寸) / 步长 ] + 1'''
    def __init__(self):
        super(Net, self).__init__()
        # 第一个卷积层，输入通道数是3（因为CIFAR-10图像是彩色的，有红绿蓝三个通道），输出通道数是6，卷积核的大小是5x5。
        # 输入是32*32*3，计算（32-5）/ 1 + 1 = 28，那么通过conv1输出的结果是28*28*6
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 最大池化层，窗口大小和步长都是2
        # 输入是28*28*6，窗口2*2，计算28 / 2 = 14，那么通过max_pool1层输出结果是14*14*6
        self.pool = nn.MaxPool2d(2, 2)
        # 第二个卷积层，输入通道数是6（来自上一层的输出），输出通道数是16，卷积核的大小是5x5。
        # 输入是14*14*6，计算（14 - 5）/ 1 + 1 = 10，那么通过conv2输出的结果是10*10*16
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 第一个全连接层，输入节点数是 16 * 5 * 5，输出节点数是 120。
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 第二个全连接层，输入节点数是 120，输出节点数是 84。
        self.fc2 = nn.Linear(120, 84)
        # 第三个全连接层，输入节点数是84，输出节点数是10。输出节点数为10是因为CIFAR-10数据集有10个类别。
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        '''定义了网络的前向传播函数。在PyTorch中，只需要定义前向传播函数，
        后向传播函数会通过自动求导机制自动生成。这个函数的输入是一个batch的图像数据，
        输出是这个batch在每个类别上的得分。'''
        # 第一层卷积，然后通过激活函数ReLU，然后进行最大池化。
        # 32x32x3 --> 28x28x6 --> 14x14x6
        x = self.pool(F.relu(self.conv1(x)))
        # 第二层卷积，然后通过激活函数ReLU，然后进行最大池化。
        # 14x14x6 --> 10x10x16 --> 5x5x16
        x = self.pool(F.relu(self.conv2(x)))
        # 将二维的特征图(featue map)展平为一维，准备输入到全连接层。
        x = x.view(-1, 16 * 5 * 5)
        # 第一个全连接层，然后通过激活函数ReLU
        x = F.relu(self.fc1(x))
        # 第二个全连接层，然后通过激活函数ReLU
        x = F.relu(self.fc2(x))
        # 第三个全连接层，输出层
        x = self.fc3(x)
        # x是网络的输出，是一个大小为（batch_size, 10）的Tensor，每一行代表一个图像在每个类别上的得分。
        return x


net = Net()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.

import torch.optim as optim
# 定义了损失函数，这里使用的是交叉熵损失。对于分类问题，交叉熵损失是一种常用的损失函数。
# 这个函数将模型的输出（即在每个类别上的得分）和真实的标签作为输入，计算出一个标量值，代表了模型的损失。
# 模型的训练目标就是最小化这个损失。
criterion = nn.CrossEntropyLoss()
# 在每一步训练中，优化器都会根据损失函数的梯度来更新模型的参数，从而减小损失。
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

########################################################################
# 4. Train the network
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 对训练数据集进行遍历，每一次循环，都会从trainloader中取出一个batch的数据。
        # get the inputs; data is a list of [inputs, labels]
        #inputs, labels = data
        # 获取一批训练数据和对应的标签，并将它们移到之前定义的设备（GPU或CPU）上。
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        # 在进行反向传播之前，需要将模型的所有参数的梯度清零。
        # 因为PyTorch的特性是累积梯度，如果不清零，梯度会被累积起来而不是被替换。
        optimizer.zero_grad()

        # forward + backward + optimize
        # 前向传播，将输入数据 inputs 传入模型，得到输出 outputs
        outputs = net(inputs)
        # 将模型的输出outputs和真实的标签labels作为输入，计算得到损失loss。
        loss = criterion(outputs, labels)
        # 反向传播，计算损失函数关于模型参数的梯度。
        loss.backward()
        # 优化器根据反向传播计算得到的梯度来更新模型的参数
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # 每2000个mini-batches打印一次平均损失
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            # 重置累积损失
            running_loss = 0.0

print('Finished Training')

########################################################################
# Let's quickly save our trained model:
# 保存模型
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

########################################################################
# See `here <https://pytorch.org/docs/stable/notes/serialization.html>`_
# for more details on saving PyTorch models.
#
# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We have trained the network for 2 passes over the training dataset.
# But we need to check if the network has learnt anything at all.
#
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
#
# Okay, first step. Let us display an image from the test set to get familiar.

dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

########################################################################
# Next, let's load back in our saved model (note: saving and re-loading the model
# wasn't necessary here, we only did it to illustrate how to do so):
# # 从本地加载模型
net = Net()
net.load_state_dict(torch.load(PATH))

########################################################################
# Okay, now let us see what the neural network thinks these examples above are:
# 在测试集samples上推理
outputs = net(images)

########################################################################
# The outputs are energies for the 10 classes.
# The higher the energy for a class, the more the network
# thinks that the image is of the particular class.
# So, let's get the index of the highest energy:

# torch.max()函数找到模型输出中每个样本得分最高的类别的索引。
# torch.max()函数会返回两个Tensor，第一个Tensor是每行的最大值，第二个Tensor是最大值所在的索引（即预测的类别）。
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

########################################################################
# The results seem pretty good.
#
# Let us look at how the network performs on the whole dataset.
# 评估模型在整个测试集上的性能，它计算了模型的预测正确率（accuracy），即预测正确的样本数量占总样本数量的比例。
correct = 0 # 正确预测的数量
total = 0 # 总的样本数量
# 在评估模型的性能时，不需要更新模型的参数，也就不需要梯度。
with torch.no_grad():
    # 对测试数据集进行遍历
    for data in testloader:
        # 获取一批测试数据和对应的标签
        images, labels = data
        outputs = net(images)
        # 找到模型输出中每个样本得分最高的类别的索引，即模型的预测结果。
        _, predicted = torch.max(outputs.data, 1)
        # 更新总的样本数量，labels.size(0)返回这个batch的样本数量。
        total += labels.size(0)
        # 更新正确预测的数量
        # 首先(predicted==labels)返回一个布尔型的Tensor，表示预测结果和真实标签是否相等。
        # 然后，使用sum()函数计算这个batch中预测正确的样本数量，最后使用item()函数将这个数量转换为Python的标量。
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' %
      (100 * correct / total))

########################################################################
# That looks way better than chance, which is 10% accuracy (randomly picking
# a class out of 10 classes).
# Seems like the network learnt something.
#
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:
# 评估模型在每个类别上的预测性能，它分别计算了模型在每个类别上的预测正确的样本数量和总的样本数量。
class_correct = list(0. for i in range(10)) # 初始化每个类别的正确预测数量
class_total = list(0. for i in range(10)) # 总的样本数量
with torch.no_grad():
    # 每一次循环，都会取出一个batch的数据。
    for data in testloader:
        # 获取一批测试数据和对应的标签
        images, labels = data
        outputs = net(images)
        # 找到模型输出中每个样本得分最高的类别的索引，即模型的预测结果。
        _, predicted = torch.max(outputs, 1)
        # (predicted==labels)返回一个布尔型的Tensor，表示预测结果和真实标签是否相等。
        # 然后，使用squeeze()函数移除这个Tensor中长度为1的维度，使其变成一个一维Tensor。
        c = (predicted == labels).squeeze()
        # 对一个batch中的四个样本进行处理
        for i in range(4):
            # 获取第i个样本的真实标签
            label = labels[i]
            # 如果模型对第i个样本的预测结果是正确的，那么c[i]的值为1，否则为0。
            # 这里将c[i]的值累加到对应类别的正确预测数量中。
            class_correct[label] += c[i].item()
            # 更新对应类别的总的样本数量
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' %
          (classes[i], 100 * class_correct[i] / class_total[i]))

########################################################################
# Okay, so what next?
#
# How do we run these neural networks on the GPU?
#
# Training on GPU
# ----------------
# Just like how you transfer a Tensor onto the GPU, you transfer the neural
# net onto the GPU.
#
# Let's first define our device as the first visible cuda device if we have
# CUDA available:

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

########################################################################
# The rest of this section assumes that ``device`` is a CUDA device.
#
# Then these methods will recursively go over all modules and convert their
# parameters and buffers to CUDA tensors:
#
# .. code:: python
#
#     net.to(device)
#
#
# Remember that you will have to send the inputs and targets at every step
# to the GPU too:
#
# .. code:: python
#
#         inputs, labels = data[0].to(device), data[1].to(device)
#
# Why dont I notice MASSIVE speedup compared to CPU? Because your network
# is really small.
#
# **Exercise:** Try increasing the width of your network (argument 2 of
# the first ``nn.Conv2d``, and argument 1 of the second ``nn.Conv2d`` –
# they need to be the same number), see what kind of speedup you get.
#
# **Goals achieved**:
#
# - Understanding PyTorch's Tensor library and neural networks at a high level.
# - Train a small neural network to classify images
#
# Training on multiple GPUs
# -------------------------
# If you want to see even more MASSIVE speedup using all of your GPUs,
# please check out :doc:`data_parallel_tutorial`.
#
# Where do I go next?
# -------------------
#
# -  :doc:`Train neural nets to play video games </intermediate/reinforcement_q_learning>`
# -  `Train a state-of-the-art ResNet network on imagenet`_
# -  `Train a face generator using Generative Adversarial Networks`_
# -  `Train a word-level language model using Recurrent LSTM networks`_
# -  `More examples`_
# -  `More tutorials`_
# -  `Discuss PyTorch on the Forums`_
# -  `Chat with other users on Slack`_
#
# .. _Train a state-of-the-art ResNet network on imagenet: https://github.com/pytorch/examples/tree/master/imagenet
# .. _Train a face generator using Generative Adversarial Networks: https://github.com/pytorch/examples/tree/master/dcgan
# .. _Train a word-level language model using Recurrent LSTM networks: https://github.com/pytorch/examples/tree/master/word_language_model
# .. _More examples: https://github.com/pytorch/examples
# .. _More tutorials: https://github.com/pytorch/tutorials
# .. _Discuss PyTorch on the Forums: https://discuss.pytorch.org/
# .. _Chat with other users on Slack: https://pytorch.slack.com/messages/beginner/

# %%%%%%INVISIBLE_CODE_BLOCK%%%%%%
del dataiter
# %%%%%%INVISIBLE_CODE_BLOCK%%%%%%
