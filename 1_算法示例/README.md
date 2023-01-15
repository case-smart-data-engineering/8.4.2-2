# 算法示例

## 使用指南

1. 按 `CTRL + P` 打开命令行面板，输入 "terminal: Create New Terminal" 打开一个命令行终端.
2. 在命令行里输入 `cd 1_算法示例` 并按 `ENTER` 进入"算法示例"目录。
3. 在命令行里输入 `python solution.py` 按 `ENTER` 运行示例程序。

该示例是使用 dgl 库中的知识图谱 FB15K237 数据集。
实体：14541
关系：237
训练边：272115
验证边：17535
测试边：20466

示例 solution.py 文件是先通过 R-GCN 实现实体分类，再将 R-GCN 生成的实体表示输入到预测层中，以预测可能的关系。
该示例 solution.py 文件会输出三元组 <subject,relation,object> ，就是本来 KG 里没有的，我们推理出来是存在的边。

在运行 solution.py 文件之前，需要先安装以下模型：
在terminal中输入：
pip install numpy
pip install torch
pip install dgl
