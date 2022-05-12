
import dgl
from dgl.nn import GraphConv
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class Encoder(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):  
        super(Encoder, self).__init__()
        self.shift_embedding = nn.Linear(5,in_feats)
        self.worker_embedding = nn.Linear(2,in_feats)
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, out_feats)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return g 


class Decoder(nn.Module):
    def __init__(self):  
        super(Decoder, self).__init__()
        self.shift_attention_embedding = nn.Linear(32,32)
        self.worker_attention_embedding = nn.Linear(32,32) 
        self.softmax = nn.Softmax(dim=0) 
          
    def forward(self, g, shift_id):
        worker_embeddings = self.worker_attention_embedding(g.ndata['h'][4:])
        shift_embedding = self.shift_attention_embedding(g.ndata['h'][shift_id])
        attention_scores = torch.inner(shift_embedding,worker_embeddings) 
        norm_scores = attention_scores / math.sqrt(32)
        return self.softmax(norm_scores)



class Policy(nn.Module):    
    def __init__(self, state, encoder, decoder):
        super(Policy, self).__init__() 
        self.state = torch.from_numpy(state).float()
        self.shift_feature_count = 5    # (num shifts - 1) + num features (2: time of day, day of week)
        self.shift_index = (torch.sum(self.state[:,self.shift_feature_count:],1) == 0).nonzero(as_tuple=True)[0][0]
        self.encoder = encoder 
        self.decoder = decoder
        self.g =  self.grapher()
        

    def grapher(self): 
        """
        Creates worker features matrix
        Creates worker and shift feature embeddings
        Adds worker feature embeddings to graph
        Adds shift feature embeddings to graph
        Creates hetergraph from shift and worker nodes
        Returns a bipartite graph
        """
        num_workers = len(self.state[0,self.shift_feature_count:])
        a = np.arange(0,num_workers,1)
        w_features = np.zeros((a.size, a.max()+1))
        w_features[np.arange(a.size),a] = 1
        t_w_features = torch.from_numpy(w_features).float()

        embedded_s = self.encoder.shift_embedding(self.state[:,:self.shift_feature_count])
        embedded_w = self.encoder.worker_embedding(t_w_features) 

        edge_tuples = []
        for i in range(len(self.state[:,0])):
            for j in range(len(self.state[0,self.shift_feature_count:])):
                edge_tuples.append((i,j))

        s_graph = dgl.heterograph(
            {('shift', '-', 'worker') : edge_tuples})
        
        s_graph.nodes['shift'].data['x'] = embedded_s
        s_graph.nodes['worker'].data['x'] = embedded_w

        hg = dgl.to_homogeneous(s_graph,['x'])
        bg = dgl.to_bidirected(hg,['x'])

        return bg 

    def forward(self):
        encoded_graph = self.encoder(self.g, self.g.ndata['x'])
        self.probs = self.decoder(encoded_graph,self.shift_index)

        return self.probs


state = np.array([[0, 0, 0, 0, 1, 1, 0],
                  [1, 0, 0, 0, 0, 0, 1],
                  [0, 1, 0, 1, 1, 1, 0],
                  [0, 0, 1, 1, 0, 0, 0]])

encoder = Encoder(4, 32, 32)
decoder = Decoder()
policy = Policy(state, encoder, decoder)

policy.forward()





