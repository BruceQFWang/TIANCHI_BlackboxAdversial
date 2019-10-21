'''
author:xingyc
我认为线上计算分时，其l2范数求解是除以向量的维度
即112*112*3，与之前我的计算相比，相当于除以根3
因为25.5*1.732=46.166，所以之前我采用了如前述的计算结果；
通过分数与自计算的l2范数，可以推算出失败的对抗样本量
'''

import sys
a = 712
d = 44.1673
# l2范数
b = float(sys.argv[1])
# score
c = float(sys.argv[2])

x = a *(c - d) / (b - d)

print('Success number is %d, fault number is %d' % (round(x), round(a - x)))