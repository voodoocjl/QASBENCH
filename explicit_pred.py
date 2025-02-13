import torch
import numpy as np
from utils import *
from models import *
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


torch.manual_seed(42)

arch_code = [4, 4]
fold = 1

nets, accs = load_data()
labels = get_label(torch.from_numpy(np.asarray(accs, dtype=np.float32).reshape(-1, 1)), 4)

X = nets
y = labels.long()  # 三分类标签

# 创建Dataset
dataset = TensorDataset(X, y)

# 划分训练集和测试集 (8:2)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 初始化模型
device = torch.device("cpu")
model = FC(arch_code).to(device)
# model = CNNWithGAP().to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
# 在优化器中加入L2正则化
# optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
# model.load_state_dict(torch.load("best_model.pth"))

# 训练参数
num_epochs = 200
best_acc = 0.0

# 训练循环
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss, train_correct, total = 0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs[0]
        loss = criterion(outputs, labels)
        loss.backward(retain_graph=True)
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        train_correct += (predicted[:, 0] == labels[:, 0]).sum().item()  # 只要 y3 是正确的，就算对
        train_loss += loss.item() * inputs.size(0)

    train_loss = train_loss / total
    train_acc = train_correct / total

    # 验证阶段
    model.eval()
    test_loss, test_correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            outputs = outputs[0]
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            test_correct += (predicted[:, 0] == labels[:, 0]).sum().item()  # 只要 y3 是正确的，就算对
            test_loss += loss.item() * inputs.size(0)

    test_loss = test_loss / total
    test_acc = test_correct / total

    # 打印结果
    print(
        f'Epoch [{epoch + 1}/{num_epochs}] \t Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}\t\t Test Loss:  {test_loss:.4f} Acc: {test_acc:.4f}',end='')
    # 保存最佳模型
    if test_acc > best_acc:
        best_acc = test_acc
        # torch.save(model.state_dict(), 'best_model.pth')
        print('\tSaving model...')
    else:
        print()

print(f'Best Test Accuracy: {best_acc:.4f}')