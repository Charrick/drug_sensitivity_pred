# encoding: utf-8
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, relation_file, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)

        mask = self.readRelationFromFile(relation_file)
        self.register_buffer('mask', mask)

        self.iter = 0

    def forward(self, input):
        masked_weight = self.weight * self.mask

		# if self.iter % 200 == 0:
		#	with open('w-{}-{}-uber.txt'.format(self.in_features, self.iter), 'w') as f:
		#		for path_idx in range(len(self.weight.data)):
		#			f.write('{}\n'.format(','.join([str(x) for x in list(masked_weight.data[path_idx])])))
        #self.iter += 1

        return F.linear(input, masked_weight, self.bias)

    def readRelationFromFile(self, relation_file):
        mask = []
        with open(relation_file, 'r') as f:
            for line in f:
                l = [int(x) for x in line.strip().split(',')]
                for item in l:
                    assert item == 1 or item == 0  # relation 只能为0或者1
                mask.append(l)
        return Variable(torch.Tensor(mask)).cuda()
