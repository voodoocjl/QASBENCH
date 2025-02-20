import torch
import numpy as np
import torch.nn.functional as F
from Arguments import Arguments

args = Arguments()

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

def load_data(rep = 'explicit'):
    fold = 1
    with open('data/MNIST_4.csv', 'r') as f:
        data = f.readlines()
        data = [d.split('"') for d in data]
        data = [(eval(single_enta)[0], eval(single_enta)[1], float(acc.replace(',', ''))) for _, single_enta, acc in data]
        nets = []
        accs = []

        if rep == 'explicit':
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
        else:
            for s ,e, a in data:
                s = np.array(s)
                datauploading = [d[[1, 3, 5, 7]] for d in s]
                rot = [d[[2, 4, 6, 8]] for d in s]
                enta = [d[1:] for d in e]
                nets.append((datauploading, rot, enta))
                accs.append(a)

    return nets, accs

def encode_gate_type():
    gate_dict = {}
    ops = args.allowed_gates.copy()
    ops.insert(0, 'START')
    ops.append('END')
    ops_len = len(ops)
    ops_index = torch.tensor(range(ops_len))
    type_onehot = F.one_hot(ops_index, num_classes=ops_len)
    for i in range(ops_len):
        gate_dict[ops[i]] = type_onehot[i]
    return gate_dict

def get_wires(op):
    if op[0] == 'C(U3)':
        return [op[1], op[2]]
    else:
        return [op[1]]

def get_gate_and_adj_matrix(circuit_list, arch_code):
    n_qubits = arch_code[0]
    gate_matrix = []
    op_list = []
    cl = list(circuit_list).copy()
    if cl[0] != 'START':
        cl.insert(0, 'START')
    if cl[-1] != 'END':
        cl.append('END')
    # cg = get_circuit_graph(circuit_list)
    gate_dict = encode_gate_type()
    gate_matrix.append(gate_dict['START'].tolist() + [1] * n_qubits)
    op_list.append('START')
    for op in circuit_list:
        op_list.append(op)
        op_qubits = [0] * n_qubits
        op_wires = get_wires(op)
        for i in op_wires:
            op_qubits[i] = 1
        op_vector = gate_dict[op[0]].tolist() + op_qubits
        gate_matrix.append(op_vector)
    gate_matrix.append(gate_dict['END'].tolist() + [1] * n_qubits)
    op_list.append('END')

    op_len = len(op_list)
    adj_matrix = np.zeros((op_len, op_len), dtype=int)
    for index, op in enumerate(circuit_list):
        ancestors = []
        target_wires = get_wires(op)
        if op[0] == 'C(U3)':
            found_wires = {target_wires[0]: False, target_wires[1]: False}
            max_ancestors = 2
        else:
            found_wires = {target_wires[0]: False}
            max_ancestors = 1

        for i in range(index - 1, -1, -1):
            op_wires = get_wires(circuit_list[i])
            if any(not found_wires[w] for w in op_wires if w in found_wires):
                ancestors.append(circuit_list[i])

                for w in op_wires:
                    if w in found_wires:
                        found_wires[w] = True
                if len(ancestors) >= max_ancestors:
                    break
        if len(ancestors) == 0:
            adj_matrix[0][op_list.index(op)] = 1
        else:
            for j in range(len(ancestors)):
                adj_matrix[op_list.index(ancestors[j])][op_list.index(op)] = 1

        descendants = []
        if op[0] == 'C(U3)':
            found_wires = {target_wires[0]: False, target_wires[1]: False}
            max_descendants = 2
        else:
            found_wires = {target_wires[0]: False}
            max_descendants = 1

        for i in range(index + 1, len(circuit_list)):
            op_wires = get_wires(circuit_list[i])
            if any(not found_wires[w] for w in op_wires if w in found_wires):
                descendants.append(circuit_list[i])
                for w in op_wires:
                    if w in found_wires:
                        found_wires[w] = True
                if len(descendants) >= max_descendants:
                    break
        if len(descendants) < max_descendants:
            adj_matrix[op_list.index(op)][op_len - 1] = 1

    return cl, gate_matrix, adj_matrix

configs = [{'GAE': # 0
                {'activation_ops':torch.sigmoid},
            'loss':
                {'loss_ops':F.mse_loss, 'loss_adj':F.mse_loss},
            'prep':
                {'method':3, 'lbd':0.5}
            },
           {'GAE': # 1
                {'activation_ops':torch.softmax},
            'loss':
                {'loss_ops':torch.nn.BCELoss(), 'loss_adj':torch.nn.BCELoss()},
            'prep':
                {'method':3, 'lbd':0.5}
            },
           {'GAE': # 2
                {'activation_ops': torch.softmax},
            'loss':
                {'loss_ops': F.mse_loss, 'loss_adj': torch.nn.BCELoss()},
            'prep':
                {'method':3, 'lbd':0.5}
            },
           {'GAE':# 3
                {'activation_ops':torch.sigmoid},
            'loss':
                {'loss_ops':F.mse_loss, 'loss_adj':F.mse_loss},
            'prep':
                {'method':4, 'lbd':1.0}
            },
           {'GAE': # 4
                {'activation_adj': torch.sigmoid, 'activation_ops': torch.softmax, 'adj_hidden_dim': 128, 'ops_hidden_dim': 128},
            'loss':
                {'loss_ops': torch.nn.BCELoss(), 'loss_adj': torch.nn.BCELoss()},
            'prep':
                {'method': 4, 'lbd': 1.0}
            },
           ]