import torch
import torch.nn as nn

class FC(nn.Module):
    def __init__(self, arch):
        super(FC, self).__init__()
        m = arch[0]  # qubit
        n = arch[1]  # layer
        self.fc01 = nn.Linear(m * n, 16)
        self.fc02 = nn.Linear(2 * m * n, 16)

        self.fc11 = nn.Linear(32, 64)
        self.fc12 = nn.Linear(64, 32)
        self.fc13 = nn.Linear(32, 32)

        self.cls1 = nn.Linear(64, 2)
        self.cls2 = nn.Linear(32, 2)
        self.cls3 = nn.Linear(32, 2)

        self.dropout = nn.Dropout(0.6)  # 添加Dropout层

    def forward(self, x):
        layer_n = x.shape[1]
        all = [i for i in range(0, layer_n)]
        topo = [i for i in range(0, layer_n, 3)]
        single = [i for i, j in enumerate(all) if i not in topo]

        x1 = x[:, topo, :]
        x2 = x[:, single, :]

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x1 = torch.relu(self.fc01(x1))
        x2 = torch.relu(self.fc02(x2))
        x = torch.cat((x1, x2), 1)

        x = self.dropout(x)  # 在适当的位置添加Dropout

        y1 = torch.relu(self.fc11(x))
        y2 = torch.relu(self.fc12(y1))
        y3 = torch.relu(self.fc13(y2))

        y1 = self.cls1(y1)
        y2 = self.cls2(y2)
        y3 = self.cls3(y3)

        preds_c1 = torch.argmax(y1, dim=1)
        preds_c2 = torch.argmax(y2, dim=1)
        preds_c3 = torch.argmax(y3, dim=1)
        preds = torch.stack((preds_c3, preds_c2, preds_c1), dim=1)

        return [torch.stack((y3, y2, y1), dim=1).transpose(1, 2), preds]
    
# 定义 CNN with GAP 模型（简化版）
class CNNWithGAP(nn.Module):
    def __init__(self):
        super(CNNWithGAP, self).__init__()
        # 第一层卷积 + 池化
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 减少通道数
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 减少通道数
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 减少通道数

        # 池化层
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1)

        # 全局平均池化层
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 全连接层
        self.fc = nn.Linear(64, 3)  # 输出类别数为3

        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        # Dropout
        self.dropout = nn.Dropout(0.8)  # 添加 Dropout

    def forward(self, x):
        x = x.unsqueeze(1)  # 增加一个通道维度，变为 (batch_size, 1, 12, 4)

        # 卷积层 + 激活函数 + Batch Normalization + 池化
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))

        # 全局平均池化
        x = self.gap(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # 全连接层 + Dropout
        x = self.dropout(x)
        x = self.fc(x)

        # 对每个类别的概率分布进行 sigmoid（多标签分类）
        x = torch.sigmoid(x)
        return x
    
class FCN(nn.Module):
    """Fully Convolutional Network"""
    def __init__(self, arch):
        super(FCN, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()

        self.fc11 = nn.Linear(1792, 64)
        # self.fc11 = nn.Linear(2240, 64)
        # self.fc11 = None
        self.fc12 = nn.Linear(64, 32)
        self.fc13 = nn.Linear(32, 32)
        
        self.cls1= nn.Linear(64,2)
        self.cls2= nn.Linear(32,2)
        self.cls3= nn.Linear(32,2)

    def forward(self, x):        

        x = x.unsqueeze(1)
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = self.flatten(x)
        
        y1 = torch.relu(self.fc11(x))
        y2 = torch.relu(self.fc12(y1))
        y3 = torch.relu(self.fc13(y2))

        y1 = self.cls1(y1)
        y2 = self.cls2(y2)
        y3 = self.cls3(y3)

        preds_c1 = torch.argmax(y1, dim=1)
        preds_c2 = torch.argmax(y2, dim=1)
        preds_c3 = torch.argmax(y3, dim=1)
        preds = torch.stack((preds_c3, preds_c2, preds_c1),dim=1)
       
        return [torch.stack((y3, y2, y1), dim=1).transpose(1,2), preds]