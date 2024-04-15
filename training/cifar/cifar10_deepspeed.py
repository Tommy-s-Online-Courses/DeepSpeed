import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import deepspeed


def add_argument():

    parser = argparse.ArgumentParser(description='CIFAR')

    #data
    # cuda
    parser.add_argument('--with_cuda',
                        default=False,
                        action='store_true',
                        help='use CPU in case there\'s no GPU support')

    # 指数移动平均EMA 是一种常见的平滑技术，主要原理是对模型的参数进行加权平均，
    # 其中近期的参数值得到的权重较大，而较早的参数值得到的权重较小。这种平滑处理可以有效减小模型在训练过程中的震荡，
    # 从而提高模型在验证集和测试集上的表现。
    parser.add_argument('--use_ema',
                        default=False,
                        action='store_true',
                        help='whether use exponential moving average')

    # train
    parser.add_argument('-b',
                        '--batch_size',
                        default=32,
                        type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e',
                        '--epochs',
                        default=30,
                        type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    parser.add_argument('--log-interval',
                        type=int,
                        default=2000,
                        help="output logging information at a given interval")

    parser.add_argument('--moe',
                        default=False,
                        action='store_true',
                        help='use deepspeed mixture of experts (moe)')

    # 在各个并行运算节点之间分配“专家”的数量
    parser.add_argument('--ep-world-size',
                        default=1,
                        type=int,
                        help='(moe) expert parallel world size')

    parser.add_argument('--num-experts',
                        type=int,
                        nargs='+',
                        default=[
                            1,
                        ],
                        help='number of experts list, MoE related.')

    # 指定在训练一个多层感知器（MLP）模型时，采用的类型。
    # 选项standard和residual可能分别表示标准的MLP模型和使用了残差连接的MLP模型。
    parser.add_argument(
        '--mlp-type',
        type=str,
        default='standard',
        help=
        'Only applicable when num-experts > 1, accepts [standard, residual]')

    parser.add_argument('--top-k',
                        default=1,
                        type=int,
                        help='(moe) gating top 1 and 2 supported')

    # 指定在使用混合专家模型时，每个“专家expert”的最小容量
    parser.add_argument(
        '--min-capacity',
        default=0,
        type=int,
        help=
        '(moe) minimum capacity of an expert regardless of the capacity_factor'
    )

    # 指定在使用混合专家模型时，采用的噪声门控策略。
    # 噪声门控可能是一种在选择“专家”时引入噪声的策略，可以提高模型的鲁棒性。
    parser.add_argument(
        '--noisy-gate-policy',
        default=None,
        type=str,
        help=
        '(moe) noisy gating (only supported with top-1). Valid values are None, RSample, and Jitter'
    )

    # 指定在使用混合专家模型时，是否创建单独的参数组。
    parser.add_argument(
        '--moe-param-group',
        default=False,
        action='store_true',
        help=
        '(moe) create separate moe param groups, required when using ZeRO w. MoE'
    )

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args

# 初始化分布式训练环境
# 初始化进程组，也就是在多个GPU上运行的进程。这包括设置每个进程的通信策略，例如它们如何共享模型参数、梯度等。
# 它的另一个工作是确定每个进程的角色，例如哪个进程是主进程，哪些进程是辅助进程。
deepspeed.init_distributed()

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].
# .. note::
#     If running on Windows and you get a BrokenPipeError, try setting
#     the num_worker of torch.utils.data.DataLoader() to 0.

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# torch.distributed.get_rank()方法返回当前进程在进程组中的rank（索引）。
# 在分布式系统中，每个进程都被分配了一个独特的rank，用于标识这个进程，Rank从0开始。
if torch.distributed.get_rank() != 0:
    # might be downloading cifar data, let rank 0 download first
    # torch.distributed.barrier()方法是一种同步机制，用于确保在所有进程中都调用了barrier之后，才能继续执行后续操作。
    # 在实际使用中，我们通常用这个方法来确保不同进程之间的同步。
    # 如果没有调用barrier，有可能会出现一些进程已经开始执行后续的操作，而另一些进程还在等待的情况。
    torch.distributed.barrier()

trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform)

if torch.distributed.get_rank() == 0:
    # cifar data is downloaded, indicate other ranks can proceed
    # 当rank为0的进程下载完数据后，它会调用一次torch.distributed.barrier()。
    # 这表示数据已经下载完毕，其他的进程可以继续执行了。
    torch.distributed.barrier()

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=32,
                                          shuffle=True,
                                          num_workers=12)

testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=32,
                                         shuffle=False,
                                         num_workers=12)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')

########################################################################
# Let us show some of the training images, for fun.

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

########################################################################
# 2. Define a Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

import torch.nn as nn
import torch.nn.functional as F

args = add_argument()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        # 否使用DeepSpeed的专家混合（Mixture of Experts）特性
        if args.moe:
            fc3 = nn.Linear(84, 84)
            # 初始化一个空列表，用于存储创建的MoE层
            self.moe_layer_list = []
            # 每个元素表示一个MoE层中专家的数量
            for n_e in args.num_experts:
                # create moe layers based on the number of experts
                self.moe_layer_list.append(
                    # MoE层是使用deepspeed.moe.layer.MoE创建
                    deepspeed.moe.layer.MoE(
                        hidden_size=84, # MoE层的隐藏单元数，即输入和输出的维度。
                        expert=fc3, # 用作MoE层中的专家的模型
                        num_experts=n_e, # MoE层中专家的数量
                        ep_size=args.ep_world_size, # 每个专家的预期大小, 这个参数用于DeepSpeed的分布式设置
                        use_residual=args.mlp_type == 'residual', # 如果args.mlp_type等于'residual'，则在MoE层中使用残差连接
                        k=args.top_k, # 在MoE层中的门控机制中，选择的顶部专家的数量
                        min_capacity=args.min_capacity, # 最小容量, 这个参数用于DeepSpeed的分布式设置
                        noisy_gate_policy=args.noisy_gate_policy)) # 噪声门策略, 这个参数用于在MoE层中添加噪声，以增强模型的稳健性
            # 将Python列表转换为PyTorch的ModuleList，这样PyTorch能够自动跟踪MoE层的参数
            self.moe_layer_list = nn.ModuleList(self.moe_layer_list)
            # 定义一个线性层（全连接层），输入维度为84，输出维度为10
            self.fc4 = nn.Linear(84, 10)
        else:
            self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 是否使用混合专家层
        if args.moe:
            # 遍历moe_layer_list中的每个MoE层
            for layer in self.moe_layer_list:
                # MoE层返回三个输出，但只有第一个输出（x）在这里被使用
                x, _, _ = layer(x)
            x = self.fc4(x)
        else:
            x = self.fc3(x)
        return x


net = Net()


def create_moe_param_groups(model):
    '''使用混合专家（Mixture of Experts，MoE）层的模型创建参数组，
    优势：在设置优化器时非常有用，特别是当模型中包含MoE层时。
    这样可以使得在训练过程中，不同的MoE层可以有不同的学习率或者其他的优化策略。'''

    # 这个函数用于将模型的参数分割到不同的MoE组中，以便用于优化器。
    from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
    # 创建一个字典，其中包含模型的所有参数和参数的名字
    parameters = {
        'params': [p for p in model.parameters()],
        'name': 'parameters'
    }
    # 将参数分割到不同的MoE组中，然后返回结果
    return split_params_into_different_moe_groups_for_optimizer(parameters)

# 获取需要梯度的参数，这些参数是用于网络训练的。
# 返回的是那些requires_grad属性为True的参数，也就是说这些参数在训练过程中需要被更新。
parameters = filter(lambda p: p.requires_grad, net.parameters())
# 否需要对模型的参数进行MoE参数组的处理
if args.moe_param_group:
    parameters = create_moe_param_groups(net)

# Initialize DeepSpeed to use the following features
# 1) Distributed model 这使得模型可以在多个设备或多个节点上进行训练，从而提高训练速度和扩展性。
# 2) Distributed data loader 这可以将数据的加载和预处理分布在多个设备或多个节点上，从而进一步提高数据处理和训练的速度。
# 3) DeepSpeed optimizer 这是DeepSpeed提供的优化器，可以更好地优化训练过程。

# model_engine（模型引擎，负责管理模型的训练和评估）
# optimizer（优化器，用于优化模型参数）
# trainloader（训练数据加载器，用于加载训练数据）
model_engine, optimizer, trainloader, __ = deepspeed.initialize(
    args=args, model=net, model_parameters=parameters, training_data=trainset)
# 是否启用了16位浮点数（FP16）训练，FP16训练可以加速训练，同时降低内存消耗。
fp16 = model_engine.fp16_enabled()
print(f'fp16={fp16}')

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#net.to(device)
########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.

for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        # model_engine.local_rank 是当前进程在所有进程中的排名，它决定了数据应该被送到哪个设备上。
        inputs, labels = data[0].to(model_engine.local_rank), data[1].to(model_engine.local_rank)
        if fp16:
            # 如果启用了FP16训练，就将输入的数据类型转换为半精度浮点数。
            inputs = inputs.half()
        # 将输入数据传入模型，获取模型的输出
        outputs = model_engine(inputs)
        # 计算模型输出和实际标签之间的损失
        loss = criterion(outputs, labels)
        # 计算损失的反向传播
        model_engine.backward(loss)
        # 更新模型的参数
        model_engine.step()

        # print statistics
        running_loss += loss.item()
        if i % args.log_interval == (
                args.log_interval -
                1):  # print every log_interval mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / args.log_interval))
            running_loss = 0.0

print('Finished Training')

########################################################################
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
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(16)))

########################################################################
# Okay, now let us see what the neural network thinks these examples above are:
if fp16:
    images = images.half()
outputs = net(images.to(model_engine.local_rank))

########################################################################
# The outputs are energies for the 10 classes.
# The higher the energy for a class, the more the network
# thinks that the image is of the particular class.
# So, let's get the index of the highest energy:
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(16)))

########################################################################
# The results seem pretty good.
#
# Let us look at how the network performs on the whole dataset.

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        if fp16:
            images = images.half()
        outputs = net(images.to(model_engine.local_rank))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(
            model_engine.local_rank)).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' %
      (100 * correct / total))

########################################################################
# That looks way better than chance, which is 10% accuracy (randomly picking
# a class out of 10 classes).
# Seems like the network learnt something.
#
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        if fp16:
            images = images.half()
        outputs = net(images.to(model_engine.local_rank))
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels.to(model_engine.local_rank)).squeeze()
        for i in range(16):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' %
          (classes[i], 100 * class_correct[i] / class_total[i]))
