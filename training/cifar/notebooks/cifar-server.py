from fastapi import FastAPI
from pydantic import BaseModel
from torchvision.transforms import transforms
import torch
import base64
import io
from PIL import Image
import uvicorn

app = FastAPI()

import torch.nn as nn
import torch.nn.functional as F

# 10个类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')

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

    
# 加载本地的模型

model = Net()

model.load_state_dict(torch.load("cifar_net.pth"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

model.eval()

# 定义输入的数据模型

class Item(BaseModel):
    img_base64: str
    
@app.post("/predict")
def predict(item: Item):
    # 对输入的base64图片进行解码和处理
    # 将base64编码的字符串解码回原始的二进制图像数据
    img = base64.b64decode(item.img_base64)
    # 将二进制图像数据转换为一个PIL的Image对象
    img = Image.open(io.BytesIO(img))
    # 定义transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32,32)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 图片预处理
    img = transform(img).unsqueeze(0).to(device)
    
    # 进行推理
    output = model(img)
    
    # 预测结果
    # pred = output.argmax(dim=1, keepdim=True) 
    pred = torch.argmax(output, dim=1, keepdim=True)
    class_name = classes[pred]


    return {"prediction" : int(pred), "class_name": class_name}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)