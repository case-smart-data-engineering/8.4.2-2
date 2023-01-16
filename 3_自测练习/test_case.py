#!/usr/bin/env python3

import os

# 测试用例
os.system('my_solution.py')
f_1 = open('out_1.txt', 'r')
data_1 = f_1.read()
f = open('out.txt', 'r')
data = f.read()
assert data == data_1  # 判断输出结果是否和预期结果一样