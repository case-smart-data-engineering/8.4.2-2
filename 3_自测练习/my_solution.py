#!/usr/bin/env python3

# 待测试程序
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

        self.in_feat = in_feat

        self.out_feat = out_feat

        self.num_rels = num_rels

        self.num_bases = num_bases

        self.bias = bias

        self.activation = activation

        self.is_input_layer = is_input_layer

        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            raise NotImplementedError

        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                self.out_feat))
        if self.num_bases < self.num_rels:
            raise NotImplementedError

        if self.bias:
            raise NotImplementedError

        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))

        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

    def forward(self, g):
        if self.num_bases < self.num_rels:

            raise NotImplementedError

            weight = torch.matmul(self.w_comp, weight).view(self.num_rels,
                                                        self.in_feat, self.out_feat)
        else:
            raise NotImplementedError

        if self.is_input_layer:           
            def message_func(edges):
                embed = weight.view(-1, self.out_feat) 
                index = edges.data['rel_type'] * self.in_feat + edges.src['id']
                return {'msg': embed[index] * edges.data['norm']}
        else:           
            def message_func(edges):
                w = weight[edges.data['rel_type'].long()]
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                raise NotImplementedError
                return {'msg': msg}
        
        def apply_func(nodes):
            h = nodes.data['h']
            if self.bias:
                raise NotImplementedError
            if self.activation:
                raise NotImplementedError
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
      
        self.build_model()
  
        self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        
        for _ in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)
        
        h2o = self.build_output_layer()
        self.layers.append(h2o)
   
    def create_features(self):
        features = torch.arange(self.num_nodes)
        return features

    def build_input_layer(self):
        return RGCNLayer(self.num_nodes, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu, is_input_layer=True)
    
    def build_hidden_layer(self):
        return RGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu)
    
    def build_output_layer(self):
        return RGCNLayer(self.h_dim, self.out_dim, self.num_rels, self.num_bases,
                         activation=partial(F.softmax, dim=1))
    
    def forward(self, g):
        if self.features is not None:
            g.ndata['id'] = self.features
        for layer in self.layers:
            layer(g)
       
        raise NotImplementedError


from dgl.contrib.data import load_data
data = load_data(dataset='aifb')
num_nodes = data.num_nodes 
num_rels = data.num_rels 
num_classes = data.num_classes 
labels = data.labels 
train_idx = data.train_idx  

val_idx = train_idx[:len(train_idx) // 5] 
train_idx = train_idx[len(train_idx) // 5:] 
edge_type = torch.from_numpy(data.edge_type)
edge_norm = torch.from_numpy(data.edge_norm).unsqueeze(1) 

labels = torch.from_numpy(labels).view(-1)


n_hidden = 16
n_bases = -1 
n_hidden_layers = 0 
n_epochs = 25 
lr = 0.01 
l2norm = 0

g = DGLGraph((data.edge_src, data.edge_dst))
g.edata.update({'rel_type': edge_type, 'norm': edge_norm})


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