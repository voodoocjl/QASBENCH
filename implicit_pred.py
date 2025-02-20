import torch
import numpy as np
from GVAE_model import preprocessing, GVAE
from utils import *
from models import *

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

torch.manual_seed(42)

arch_code = [4, 4]
fold = 1

input_dim = 2 + 5 + int(arch_code[0]/fold)
GVAE_model = GVAE((input_dim, 32, 64, 128, 64, 32, 16), normalize=True, dropout=0.3, **configs[4]['GAE'])
checkpoint = torch.load('pretrained/best_model.pt', map_location=torch.device('cpu'))
GVAE_model.load_state_dict(checkpoint)

def GVAE_translator(data_uploading, rot, enta, arch_code):
    single_list = []
    enta_list = []
    n_qubits = arch_code[0]
    n_layers = arch_code[1]

    for i in range(0, n_layers):
        single_item = []
        for j in range(0, n_qubits):
            d = int(data_uploading[i][j])
            r = int(rot[i][j])
            combination = f'{d}{r}'
            if combination == '00':
                single_item.append(('Identity', j))
            elif combination == '01':
                angle = np.random.uniform(0, 2 * np.pi)
                single_item.append(('RX', j, angle))
            elif combination == '10':
                angle = np.random.uniform(0, 2 * np.pi)
                single_item.append(('RY', j, angle))
            elif combination == '11':
                angle = np.random.uniform(0, 2 * np.pi)
                single_item.append(('RZ', j, angle))
        single_list.append(single_item)

        enta_item = []
        for j, et in enumerate(enta[i]):
            if j == int(et) - 1:
                enta_item.append(('Identity', j))
            else:
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, 2 * np.pi)
                delta = np.random.uniform(0, 2 * np.pi)
                enta_item.append(('C(U3)', j, int(et) - 1, theta, phi, delta))
        enta_list.append(enta_item)

    circuit_ops = []
    for layer in range(0, n_layers):
        circuit_ops.extend(single_list[layer])
        circuit_ops.extend(enta_list[layer])

    return circuit_ops

def arch_to_z(nets):
    adj_list, op_list = [], []
    for data_uploading, rot, enta in nets:
        circuit_ops = GVAE_translator(data_uploading, rot, enta, arch_code)
        _, gate_matrix, adj_matrix = get_gate_and_adj_matrix(circuit_ops, arch_code)
        ops = torch.tensor(gate_matrix, dtype=torch.float32).unsqueeze(0)
        adj = torch.tensor(adj_matrix, dtype=torch.float32).unsqueeze(0)
        adj_list.append(adj)
        op_list.append(ops)

    adj = torch.cat(adj_list, dim=0)
    ops = torch.cat(op_list, dim=0)
    adj, ops, prep_reverse = preprocessing(adj, ops, **configs[4]['prep'])
    encoder = GVAE_model.encoder
    encoder.eval()
    mu, logvar = encoder(ops, adj)
    return mu


nets, accs = load_data('implicit')
mu = arch_to_z(nets)
labels = get_label(torch.from_numpy(np.asarray(accs, dtype=np.float32).reshape(-1, 1)), 4)

X = mu
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
model = FCN(arch_code).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# model.load_state_dict(torch.load("best_model.pth"))

# 训练参数
num_epochs = 100
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
        torch.save(model.state_dict(), 'best_model.pth')
        print('\tSaving model...')
    else:
        print()

print(f'Best Test Accuracy: {best_acc:.4f}')