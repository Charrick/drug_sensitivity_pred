# coding: utf-8
from math import sqrt


class cacul_r(object):
    def __init__(self, pre_value, true_value):
        self.pre_value = pre_value
        self.true_value = true_value

    def multipl(self, a, b):
        sumofab = 0.0
        for i in range(len(a)):
            temp = a[i] * b[i]
            sumofab += temp
        return sumofab

    def cacul(self):
        n = len(self.pre_value)
        # 求和
        sum1 = sum(self.pre_value)
        # print sum1
        sum2 = sum(self.true_value)

        # print sum2
        # 求乘积之和
        sumofxy = self.multipl(self.pre_value, self.true_value)
        # 求平方和
        sumofx2 = sum([pow(i, 2) for i in self.pre_value])
        sumofy2 = sum([pow(j, 2) for j in self.true_value])
        num = sumofxy - sum1 * sum2 / n
        # print sumofx2, sumofy2
        # 计算皮尔逊相关系数
        den = sqrt((sumofx2 - sum1 ** 2 / n) * (sumofy2 - sum2 ** 2 / n))
        # print num, den
		if den == 0:
			pcc = 0
		else:
			pcc = num/den
        return pcc
