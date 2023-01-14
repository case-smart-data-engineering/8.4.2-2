#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from functools import partial


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(RGCNLayer, self).__init__()
        # 输入的特征维度
        self.in_feat = in_feat
        # 输出的特征维度
        self.out_feat = out_feat
        # 关系的数量
        self.num_rels = num_rels
        # 基分解中W_r分解的数量，即B的大小
        self.num_bases = num_bases
        # 是否带偏置b
        self.bias = bias
        # 激活函数
        self.activation = activation
        # 是否是输入层
        self.is_input_layer = is_input_layer

        # 如果说没设定W_r的个数（B）或者W_r的个数比关系数大，那么就直接取关系数
        # 因为这个正则化就是为了解决关系数过多而导致训练参数过多及过拟合的问题的
        # 如果没有正则化优化正常来说有几个关系就对应几个W_r
        # 因为因此肯定是B <= num_rels的
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # 创建基分解矩阵num_bases * in_feat * out_feat， 对应公式3
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                self.out_feat))
        if self.num_bases < self.num_rels:
            # 如果B小于关系数，那么就需要进行一个维度变换，
            # 关系数变成成B的数量才可以进行后续计算
            # 对应公式3
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        # 如果需要偏置则添加偏置b
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))

        # 初始化参数
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))
        # 初始化参数，如果B<num_rels说明self.w_comp需要使用，
        # 因此需要初始化，否则如果b = num_rels则不必使用这个来转化
        # 那也就用不到这个矩阵了
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))
        # 使用偏置的话也初始化
        if self.bias:
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

    def forward(self, g):
        # 如果B < 关系数
        # 那么参数矩阵就需要转换一下
        if self.num_bases < self.num_rels:
            # 根据公式3转换权重
            # weight的维度 = num_bases * in_feat * out_feat --> in_feat * num_bases * out_feat
            weight = self.weight.view(self.in_feat, self.num_bases, self.out_feat)
            # w_comp的维度 => num_rels * num_bases
            # torch.matmul(self.w_comp, weight)
            # w_comp(num_rels * num_bases)  weight(in_feat * num_bases * out_feat)
            #                             ||
            #                             V
            #              in_feat * num_rels * out_feat
            # 再经过view操作
            # weight --> num_rels * in_feat * out_feat
            weight = torch.matmul(self.w_comp, weight).view(self.num_rels,
                                                        self.in_feat, self.out_feat)
        else:
            # 如果没有正则化，即直接取所有关系，原本初始化就是这个形状
            # weight = num_rels * in_feat * out_feat
            weight = self.weight

        if self.is_input_layer:
            # 如果是输入层，需要获得节点的embedding表达
            def message_func(edges):
                embed = weight.view(-1, self.out_feat) 
                index = edges.data['rel_type'] * self.in_feat + edges.src['id']
                # edges.data['rel_type'] * self.in_feat + edges.src['id']就是取自身节点对应的关系的那种表达
                return {'msg': embed[index] * edges.data['norm']}
        else:
            # 如果不是输入层那么用计算出 邻居特征*关系 的特征值
            def message_func(edges):
                # 取出对应关系的权重矩阵
                w = weight[edges.data['rel_type'].long()]
                # 矩阵乘法获取每条边需要传递的特征msg:(65439 * 4)
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                msg = msg * edges.data['norm']
                return {'msg': msg}
        # msg求和作为节点特征
        # 有偏置加偏置。
        # 有激活加激活，主要是用于输出层设置的
        def apply_func(nodes):
            h = nodes.data['h']
            if self.bias:
                h = h + self.bias
            if self.activation:
                h = self.activation(h)
            return {'h': h}

        g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)


class Model(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels,
                 num_bases=-1, num_hidden_layers=1):
        super(Model, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers

        # 创建R-GCN层
        self.build_model()

        # 获取特征
        self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        # 输入层
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        # 隐藏层
        for _ in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)
        # 输出层
        h2o = self.build_output_layer()
        self.layers.append(h2o)

    # 初始胡化每个节点的特征
    def create_features(self):
        features = torch.arange(self.num_nodes)
        return features

    # 构建输入层
    def build_input_layer(self):
        return RGCNLayer(self.num_nodes, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu, is_input_layer=True)

    # 构建隐藏层
    def build_hidden_layer(self):
        return RGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu)
    # 构建输出层
    def build_output_layer(self):
        return RGCNLayer(self.h_dim, self.out_dim, self.num_rels, self.num_bases,
                         activation=partial(F.softmax, dim=1))
    # 前向传播
    def forward(self, g):
        if self.features is not None:
            g.ndata['id'] = self.features
        for layer in self.layers:
            layer(g)
        # 取出每个节点的隐藏层并且删除"h"特征，方便下一次进行训练
        return g.ndata.pop('h')


from dgl.contrib.data import load_data
data = load_data(dataset='aifb')
num_nodes = data.num_nodes  # 节点数量
num_rels = data.num_rels # 关系数量
num_classes = data.num_classes # 分类的类别数
labels = data.labels # 标签
train_idx = data.train_idx  # 训练集节点的index
# split training and validation set
val_idx = train_idx[:len(train_idx) // 5] # 划分验证集
train_idx = train_idx[len(train_idx) // 5:] # 划分训练集
edge_type = torch.from_numpy(data.edge_type) # 获取边的类型
edge_norm = torch.from_numpy(data.edge_norm).unsqueeze(1) # 获取边的标准化因子

labels = torch.from_numpy(labels).view(-1)


# configurations
n_hidden = 16 # 每层的神经元个数
n_bases = -1 # 直接用所有的关系，不正则化
n_hidden_layers = 0 # 使用一层输入一层输出，不用隐藏层
n_epochs = 25 # 训练次数
lr = 0.01 # 学习率
l2norm = 0
# 创建图
g = DGLGraph((data.edge_src, data.edge_dst))
g.edata.update({'rel_type': edge_type, 'norm': edge_norm})

# 创建模型
model = Model(g.num_nodes(),
              n_hidden,
              num_classes,
              num_rels,
              num_bases=n_bases,
              num_hidden_layers=n_hidden_layers)


optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2norm)

print("start training...")
model.train()
for epoch in range(n_epochs):
    optimizer.zero_grad()
    logits = model.forward(g)
    loss = F.cross_entropy(logits[train_idx], labels[train_idx].long())
    loss.backward()

    optimizer.step()

    train_acc = torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx])
    train_acc = train_acc.item() / len(train_idx)
    val_loss = F.cross_entropy(logits[val_idx], labels[val_idx].long())
    val_acc = torch.sum(logits[val_idx].argmax(dim=1) == labels[val_idx])
    val_acc = val_acc.item() / len(val_idx)
    print("Epoch {:05d} | ".format(epoch) +
          "Train Accuracy: {:.4f} | Train Loss: {:.4f} | ".format(
              train_acc, loss.item()) +
          "Validation Accuracy: {:.4f} | Validation loss: {:.4f}".format(
              val_acc, val_loss.item()))