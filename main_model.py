# coding:utf-8
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time

num_cellline_genes = 537
num_drug_genes = 741
num_pathways = 323
TINY = 1e-15
TRAIN_NUM = 197957
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
time_start=time.time()

# from SYLMaskedLinear import MaskLinearSYL
from MaskedLinear import MaskedLinear
from cacul_r import cacul_r
torch.manual_seed(100)

# �������ݼ�������data_loader
class DrugCellAUC(Dataset):
    """Drug Disease Interaction dataset."""
    def __init__(self, file_name, transform=None):
        self.pairs = []
        self.aucs = []

        mostline = 0
        with open(file_name, 'r') as f:
            for line in f:
                mostline += 1
                items = line.strip().split(',')
                cellLineId, drugId, auc, genes = items[0], items[1], items[2], items[3:]
                cell_drug_gene = [float(x) for x in genes[:num_cellline_genes + num_drug_genes]]
                assert len(cell_drug_gene) == num_cellline_genes + num_drug_genes
                cell_drug_gene = torch.Tensor(cell_drug_gene).cuda()
                self.pairs.append(cell_drug_gene)
                self.aucs.append(float(auc))
                if mostline == TRAIN_NUM:
                    break
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
        self.cell_drugMaskedFC = MaskedLinear(num_cellline_genes+ num_drug_genes, num_pathways, 'land_feature/relation.csv').cuda()
        self.lin1 = nn.Linear(num_pathways , 256).cuda()
        self.lin1_2 = nn.Linear(256, 256).cuda()
        #self.lin1_3 = nn.Linear(256, 256).cuda()
        #self.lin1_4 = nn.Linear(256, 256).cuda()
        self.lin2 = nn.Linear(256, 1).cuda()

    def forward(self, cell_drug_genes):
        # pathways = F.dropout(self.cell_drugMaskedFC(cell_drug_genes), p = 0.2, training=self.training)
        # pathways = F.relu(pathways)
        pathways = F.relu(self.cell_drugMaskedFC(cell_drug_genes))

        x = F.dropout(self.lin1(pathways), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin1_2(x), p=0.2, training=self.training)
        x = F.relu(x)
        # x = F.dropout(self.lin1_3(x), p=0.2, training=self.training)
	    # x = F.relu(x)
	    # x = F.dropout(self.lin1_4(x), p=0.2, training=self.training)
	    # x = F.relu(x)

        x = self.lin2(x)
        return x

class test_Net(nn.Module):
    def __init__(self):
        super(test_Net, self).__init__()
        self.cell_drugMaskedFC = MaskedLinear(num_cellline_genes + num_drug_genes, num_pathways, 'land_feature/relation.csv').cuda()
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

if __name__ == '__main__':
    print('run')
    batch_size = 512
    learning_rate = 0.0001
    type = 'drug'
    ID = '41'
    # train_dataset = DrugCellAUC('set/train_set3.csv')
    train_dataset = DrugCellAUC('check_data/{}ID/_train_{}{}.csv'.format(type, type, ID))
    # train_dataset = DrugCellAUC('land_feature/nor_feature.csv')
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = DNN().cuda()
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    optimizer_sgd = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0001)

    MSE_loss = []
    model.train()

    for epoch in range(550):
        _loss = 0
        outs = []
        aucs = []
        if epoch > 499:
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
            cacul_rer = cacul_r(outs,aucs)
            r ="{0:.4}".format(cacul_rer.cacul())
            epoch_loss = "{0:.4}".format(_loss.data[0]/batch_num)
            print ('epoch{},eachloss{},_r{}'.format(epoch,epoch_loss,r))
            train_mseloss = epoch_loss
            train_r = r
	    MSE_loss.append(train_mseloss)
    with open('MSELoss_{}_{}.txt'.format(batch_size,learning_rate),'w') as f:
	    for eachloss in range(len(MSE_loss)):
	        f.write('{}\n'.format(MSE_loss[eachloss]))

    #��ѧ�õĲ������ص���������
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

    # torch.save(model, 'model.pkl') #��������������
    # test_model = torch.load('model.pkl').cuda()
    # torch.save(model.state_dict(), 'model_params.pkl')
    # test_model.load_state_dict(torch.load('model_params.pkl'))

    # test_dataset = DrugCellAUC('set/test_set3.csv')
    print 'test start'
    test_dataset = DrugCellAUC('check_data/{}ID/_test_{}{}.csv'.format(type, type, ID))
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
    cacul_rer = cacul_r(outs,aucs)
    r ="{0:.4}".format(cacul_rer.cacul())
    test_r = r
    test_mseloss = "{0:.4}".format(_loss.data[0]/batch_num)
    print ('eachloss{},_r{}'.format(test_mseloss,r))
    with open ('out/{}.csv'.format(type), 'a') as f:
        f.write('{},{},{},{},{}\n'.format(ID, train_mseloss, test_mseloss, train_r, test_r))
    # ���Ԥ���outֵ
    with open('out/pre_out_{}/ID_1{}.txt'.format(type, ID),'w') as f:
        for eachout in range(len(outs)):
            f.write('{}\n'.format(outs[eachout]))
    time_end = time.time()
    print('time cost', time_end - time_start,'s')