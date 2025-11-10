import torch
from torch import nn #神经网络
import torchvision 
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# 搭建神经网络
class VisionNet(nn.Module):
    def __init__(self):
        super(VisionNet, self).__init__()
        self.model = nn.Sequential(
            # 第一层，输入3通道，输出64通道，5*5卷积核，步长1，填充2
            nn.Conv2d(3, 64, 5, 1, 2),
            nn.BatchNorm2d(64),  # 批量归一化
            nn.ReLU(),  # 非线性激活函数
            nn.MaxPool2d(2),

            # 第二层
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 第三层
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 第四层
            nn.Conv2d(128, 128, 5, 1, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            #全连接层
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  # 添加Dropout层，丢弃率为0.5
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 简单验证模型结构
if __name__ == '__main__':
    model = VisionNet()
    input = torch.ones((64, 3, 32, 32))
    output = model(input)
    print(output.shape)