
import dgl
from dgl.nn import GINConv
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import math


class Encoder(nn.Module):
    def __init__(self, shift_features, count_workers, in_feats, h_feats, out_feats):  
        super(Encoder, self).__init__()
        self.shift_features = shift_features
        self.shift_embedding = nn.Linear(self.shift_features,in_feats)
        self.worker_embedding = nn.Linear(count_workers,in_feats)
        lin = nn.Linear(in_feats, h_feats)
        lin2 = nn.Linear(h_feats, out_feats)
        self.conv1 = GINConv(lin,aggregator_type='mean')
        self.conv2 = GINConv(lin2,aggregator_type='mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return g 


class Decoder(nn.Module):
    def __init__(self, count_shifts):  
        super(Decoder, self).__init__()
        self.shift_attention_embedding = nn.Linear(32,32)
        self.worker_attention_embedding = nn.Linear(32,32) 
        self.softmax = nn.Softmax(dim=0) 
        self.count_shifts = count_shifts
          
    def forward(self, g, shift_id):
        worker_embeddings = self.worker_attention_embedding(g.ndata['h'][self.count_shifts:])
        shift_embedding = self.shift_attention_embedding(g.ndata['h'][shift_id])
        attention_scores = torch.inner(shift_embedding,worker_embeddings) 
        norm_scores = attention_scores / math.sqrt(32)
        return self.softmax(norm_scores)


class Policy(nn.Module):    
    def __init__(self, encoder, decoder):
        super(Policy, self).__init__() 
        self.encoder = encoder 
        self.decoder = decoder

    def grapher(self, state): 
        """
        Creates worker features matrix
        Creates worker and shift feature embeddings
        Adds worker feature embeddings to graph
        Adds shift feature embeddings to graph
        Creates hetergraph from shift and worker nodes
        Returns a bipartite graph
        """

        state = torch.from_numpy(state).float()
        shift_feature_count = self.encoder.shift_features

        num_workers = len(state[0,shift_feature_count:])  # (num shifts - 1) + num features (2: time of day, day of week)
        a = np.arange(0,num_workers,1)
        w_features = np.zeros((a.size, a.max()+1))
        w_features[np.arange(a.size),a] = 1
        t_w_features = torch.from_numpy(w_features).float()

        embedded_s = self.encoder.shift_embedding(state[:,:shift_feature_count])
        embedded_w = self.encoder.worker_embedding(t_w_features) 


        edge_tuples = []
        for i in range(len(state[:,0])):
            for j in range(len(state[0,shift_feature_count:])):
                edge_tuples.append((i,j))

        s_graph = dgl.heterograph(
            {('shift', '-', 'worker') : edge_tuples})
        
        s_graph.nodes['shift'].data['x'] = embedded_s
        s_graph.nodes['worker'].data['x'] = embedded_w

        hg = dgl.to_homogeneous(s_graph,['x'])
        bg = dgl.to_bidirected(hg,['x'])

        shift_index = (torch.sum(state[:,shift_feature_count:],1) == 0).nonzero(as_tuple=True)[0][0]

        return bg, shift_index

    def forward(self, state):
        graph, shift_index = self.grapher(state)
        encoded_graph = self.encoder(graph, graph.ndata['x'])
        self.probs = self.decoder(encoded_graph, shift_index)

        return self.probs
    
    def act(self, state):
        probs = self.forward(state)

        model = Categorical(probs)

        action = model.sample()
        return action.item(), model.log_prob(action)




