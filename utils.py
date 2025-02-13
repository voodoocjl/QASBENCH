import torch
import numpy as np

def qubit_fold(jobs, phase, fold=1):
    if fold > 1:
        job_list = []
        for job in jobs:
            q = job[0]
            if phase == 0:
                job_list.append([2*q] + job[1:])
                job_list.append([2*q-1] + job[1:])
            else:
                job_1 = [2*q]
                job_2 = [2*q-1]
                for k in job[1:]:
                    if q < k:
                        job_1.append(2*k)
                        job_2.append(2*k-1)
                    elif q > k:
                        job_1.append(2*k-1)
                        job_2.append(2*k)
                    else:
                        job_1.append(2*q)
                        job_2.append(2*q-1)
                job_list.append(job_1)
                job_list.append(job_2)
    else:
        job_list = jobs
    return job_list

def gen_arch(change_code, base_code):        # start from 1, not 0
    # arch_code = base_code[1:] * base_code[0]
    n_qubits = base_code[0]
    arch_code = ([i for i in range(2, n_qubits+1, 1)] + [1]) * base_code[1]
    if change_code != None:
        if type(change_code[0]) != type([]):
            change_code = [change_code]

        for i in range(len(change_code)):
            q = change_code[i][0]  # the qubit changed
            for id, t in enumerate(change_code[i][1:]):
                arch_code[q - 1 + id * n_qubits] = t
    return arch_code

def cir_to_matrix(x, y, arch_code, fold=1):
    x = qubit_fold(x, 0, fold)
    y = qubit_fold(y, 1, fold)
    qubits = arch_code[0]
    layers = arch_code[1]
    entangle = gen_arch(y, arch_code)
    entangle = np.array([entangle]).reshape(layers, qubits).transpose(1,0)
    single = np.ones((qubits, 2*layers))
    # [[1,1,1,1]
    #  [2,2,2,2]
    #  [3,3,3,3]
    #  [0,0,0,0]]

    if x != None:
        if type(x[0]) != type([]):
            x = [x]
        x = np.array(x)
        index = x[:, 0] - 1
        index = [int(index[i]) for i in range(len(index))]
        single[index] = x[:, 1:]
    arch = np.insert(single, [(2 * i) for i in range(1, layers+1)], entangle, axis=1)
    return arch.transpose(1, 0)

def normalize(x):
    try:
        x = (x - torch.mean(x, dim=(1,2)).unsqueeze(-1).unsqueeze(-1)) / torch.std(x, dim=(1,2)).unsqueeze(-1).unsqueeze(-1)
        # x = (x - torch.mean(x)) / torch.std(x)
    except Exception as e:
        x = x
    return x

def get_label(energy, tree_height, mean = None):
    # label = energy.clone()
    # if mean and (mean < float('inf')):
    #     energy_mean = mean
    # else:
    #     energy_mean = energy.mean()
    # for i in range(energy.shape[0]):
    #     label[i] = energy[i] > energy_mean

    x = energy    
    a = [[i for i in range(len(x))]]
    for i in range(1,tree_height):
        t = []
        for j in range(2**(i-1)):        
            index = a[j]
            if len(index):
                mean = x[index].mean()
            else:
                mean = []
            t.append(torch.tensor([item for item in index if x[item] >= mean]))
            t.append(torch.tensor([item for item in index if x[item] < mean]))
        a = t
    label = torch.zeros((len(x), tree_height-1))
    for i in range(len(a)):
        index = a[i]
        if len(index):
            for j in range(len(index)):
                string_num = bin(i)[2:].zfill(tree_height-1)
                label[index[j]] = torch.tensor([int(char) for char in string_num])
    return label

def load_data():
    fold = 1
    with open('data/MNIST_4.csv', 'r') as f:
        data = f.readlines()
        data = [d.split('"') for d in data]
        data = [(eval(single_enta)[0], eval(single_enta)[1], float(acc.replace(',', ''))) for _, single_enta, acc in data]
        nets = []
        accs = []

        for i in range(len(data)):
            single = data[i][0]
            enta = data[i][1]
            acc = data[i][2]
            arch_code = [len(enta), len(enta[0])-1]
            arch = cir_to_matrix(single, enta, arch_code, fold)
            nets.append(arch)
            accs.append(acc)

        nets = torch.from_numpy(np.asarray(nets, dtype=np.float32))
        nets = normalize(nets)

    return nets, accs