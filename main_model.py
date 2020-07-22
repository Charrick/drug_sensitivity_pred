# coding:utf-8
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import random
from MaskedLinear import MaskedLinear
from cacul_r import cacul_r

num_cellline_genes = 537
num_drug_genes = 741
num_pathways = 323
TINY = 1e-15
MOST_NUM = 197957
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
time_start=time.time()

# from SYLMaskedLinear import MaskLinearSYL
torch.manual_seed(100)

# �������ݼ�������data_loader
class DrugCellAUC(Dataset):
    """Drug Disease Interaction dataset."""
    def __init__(self, data_set, transform=None):
        self.pairs = []
        self.aucs = []

        set_length = len(data_set)
        for i in range(set_length):
            auc = data_set[i]['lable']
            cell_drug_gene = data_set[i]['features']
            cell_drug_gene = [float(x) for x in cell_drug_gene[:num_cellline_genes + num_drug_genes]]
            assert len(cell_drug_gene) == num_cellline_genes + num_drug_genes
            cell_drug_gene = torch.Tensor(cell_drug_gene).cuda()
            self.pairs.append(cell_drug_gene)
            self.aucs.append(float(auc))

        self.aucs = torch.FloatTensor(self.aucs).cuda()


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx = idx % len(self)
        cell_drug_gene = self.pairs[idx]
        auc = self.aucs[idx]
        return cell_drug_gene, auc


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.cell_drugMaskedFC = MaskedLinear(num_cellline_genes+ num_drug_genes, num_pathways, 'data/relation.csv').cuda()
        self.lin1 = nn.Linear(num_pathways , 256).cuda()
        self.lin1_2 = nn.Linear(256, 256).cuda()
        self.lin2 = nn.Linear(256, 1).cuda()

    def forward(self, cell_drug_genes):
        # pathways = F.dropout(self.cell_drugMaskedFC(cell_drug_genes), p = 0.2, training=self.training)
        # pathways = F.relu(pathways)
        pathways = F.relu(self.cell_drugMaskedFC(cell_drug_genes))

        x = F.dropout(self.lin1(pathways), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin1_2(x), p=0.2, training=self.training)
        x = F.relu(x)

        x = self.lin2(x)
        return x

class test_Net(nn.Module):
    def __init__(self):
        super(test_Net, self).__init__()
        self.cell_drugMaskedFC = MaskedLinear(num_cellline_genes + num_drug_genes, num_pathways, 'data/relation.csv').cuda()
        self.lin1 = nn.Linear(num_pathways, 256).cuda()
        self.lin1_2 = nn.Linear(256, 256).cuda()
        # self.lin1_3 = nn.Linear(256, 256).cuda()
        # self.lin1_4 = nn.Linear(256, 256).cuda()
        self.lin2 = nn.Linear(256, 1).cuda()

    def forward(self, cell_drug_genes):
        pathways = F.relu(self.cell_drugMaskedFC(cell_drug_genes))
        x = F.relu(self.lin1(pathways))
        x = F.relu(self.lin1_2(x))
        x = self.lin2(x)
        return x


def readfile(filename):
    mostline = 0
    all_data = []
    with open(filename, 'r') as f:
        for line in f:
            mostline += 1
            each_data = {}
            items = line.strip().split(',')
            cellLineId, drugId, auc, genes = items[0], items[1], items[2], items[3:]
            each_data['lable'] = auc
            each_data['features'] = genes
            all_data.append(each_data)
            if mostline == MOST_NUM:
                break
    f.close()
    return all_data


def main_model(train_set, test_set):
    batch_size = 512
    learning_rate = 0.0001
    train_dataset = DrugCellAUC(train_set)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = DNN().cuda()
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    optimizer_sgd = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0001)
    # MSE_loss = []
    model.train()

    for epoch in range(1, 550):
        _loss = 0
        outs = []
        aucs = []
        if epoch > 500:
            optimizer = optimizer_sgd
        for batch_idx, (cell_drug_genes, auc) in enumerate(train_data_loader):
            cell_drug_genes = Variable(cell_drug_genes).cuda()
            auc = Variable(auc.float()).cuda()
            batch_num = batch_idx + 1
            out = model(cell_drug_genes)
            optimizer.zero_grad()
            loss = criterion(out + TINY, auc)
            if epoch % 10 == 0:
                _out = torch.squeeze(out)
                for item in range(len(auc)):
                    outs.append(_out.data[item])
                    aucs.append(auc.data[item])
                _loss += loss
            # print('batch_idx {}, loss {:.4}'.format(batch_idx, loss.data[0]))
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            cacul_rer = cacul_r(outs, aucs)
            r = "{0:.4}".format(cacul_rer.cacul())
            epoch_loss = "{0:.4}".format(_loss.data[0] / batch_num)
            # print ('epoch{},eachloss{},_r{}'.format(epoch, epoch_loss, r))
        train_mseloss = epoch_loss
        train_r = r
        # MSE_loss.append(train_mseloss)

    #test start
    test_model = test_Net().cuda()
    params = model.state_dict()
    test_model.cell_drugMaskedFC.weight = torch.nn.Parameter(params['cell_drugMaskedFC.weight'])
    test_model.cell_drugMaskedFC.bias = torch.nn.Parameter(params['cell_drugMaskedFC.bias'])
    test_model.lin1.weight = torch.nn.Parameter(params['lin1.weight'])
    test_model.lin1.bias = torch.nn.Parameter(params['lin1.bias'])
    test_model.lin1_2.weight = torch.nn.Parameter(params['lin1_2.weight'])
    test_model.lin1_2.bias = torch.nn.Parameter(params['lin1_2.bias'])
    test_model.lin2.weight = torch.nn.Parameter(params['lin2.weight'])
    test_model.lin2.bias = torch.nn.Parameter(params['lin2.bias'])

    test_dataset = DrugCellAUC(test_set)
    batch_size = 128
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    _loss = 0
    outs = []
    aucs = []
    for batch_idx, (cell_drug_genes, auc) in enumerate(test_data_loader):
        cell_drug_genes = Variable(cell_drug_genes).cuda()
        auc = Variable(auc.float()).cuda()
        batch_num = batch_idx + 1
        out = test_model(cell_drug_genes)
        _out = torch.squeeze(out)
        for item in range(len(auc)):
            outs.append(_out.data[item])
            aucs.append(auc.data[item])
        loss = criterion(out + TINY, auc)
        _loss += loss
        # print('batch_idx {}, loss {:.4}'.format(batch_idx, loss.data[0]))
    cacul_rer = cacul_r(outs, aucs)
    r = "{0:.4}".format(cacul_rer.cacul())
    test_r = r
    test_mseloss = "{0:.4}".format(_loss.data[0] / batch_num)
    return test_mseloss, test_r


def validation(k_fold=10):
    kf = KFold(n_splits=k_fold)
    data = readfile('data/nor_feature.csv')
    data_arr = np.array(data)
    mse_list = []
    r_list = []
    n = 1
    for train_index, test_index in kf.split(data_arr):
        train_set = data_arr[train_index].tolist()
        test_set = data_arr[test_index].tolist()
        test_mseloss, test_r = main_model(train_set, test_set)
        mse_list.append(test_mseloss)
        r_list.append(test_r)
        print('done {}'.format(n))
        n += 1
    mse = np.mean(np.array(mse_list))
    r = np.mean(np.array(r_list))
    print('{}_fold_cv_result, mse:{}, pcc:{}'.format(k_fold, mse, r))


if __name__ == '__main__':
    print('run')
    validation(10)
    time_end = time.time()
    print('time cost', time_end - time_start,'s')