
import dgl
from dgl.nn import RelGraphConv 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import math


class Encoder(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):  
        super(Encoder, self).__init__()
        #self.shift_features = shift_features
        self.shift_embedding = nn.Linear(5,in_feats)
        #self.worker_embedding = nn.Linear(count_workers,in_feats)
        #lin = nn.Linear(in_feats, h_feats)
        #lin2 = nn.Linear(h_feats, out_feats)
        self.conv1 = RelGraphConv(in_feats, h_feats, num_rels=2)
        self.conv2 = RelGraphConv(h_feats, out_feats, num_rels=2)      

    def forward(self, g, feat, etype):
        h = self.conv1(g, feat, etype)
        h = F.relu(h)
        h = self.conv2(g, h, etype)
        #print(h)
        g.ndata['h'] = h
        return g 


class Decoder(nn.Module):
    def __init__(self):  
        super(Decoder, self).__init__()
        self.shift_attention_embedding = nn.Linear(32,32)
        self.worker_attention_embedding = nn.Linear(32,32) 
        self.softmax = nn.Softmax(dim=0) 
        #self.count_shifts = count_shifts
          
    def forward(self, g, shift_id, count_shifts):
        worker_embeddings = self.worker_attention_embedding(g.ndata['h'][count_shifts:])
        shift_embedding = self.shift_attention_embedding(g.ndata['h'][shift_id])
        attention_scores = torch.inner(shift_embedding,worker_embeddings) 
        norm_scores = attention_scores / math.sqrt(32)
        return self.softmax(norm_scores)


class Policy(nn.Module):    
    def __init__(self, encoder, decoder):
        super(Policy, self).__init__() 
        self.encoder = encoder 
        self.decoder = decoder

    def grapher(self, state, shift_features): 
        """
        Creates worker features matrix
        Creates worker and shift feature embeddings
        Adds worker feature embeddings to graph
        Adds shift feature embeddings to graph
        Creates hetergraph from shift and worker nodes
        Returns a bipartite graph
        """

        state = torch.from_numpy(state).float()
        shift_feature_count = shift_features

        num_workers = len(state[0,shift_feature_count:])  # (num shifts - 1) + num features (2: time of day, day of week)
        num_shifts = len(state)
        
        # determine where the 5 shift features start and end
        sf_start = state.shape[0] - 1
        sf_end = state.shape[1] - num_workers
        
        embedded_s = self.encoder.shift_embedding(state[:,sf_start:sf_end])
        embedded_w = torch.zeros(num_workers,32)

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

        edges = (num_shifts * num_workers) * 2
        #print(edges)
        bg.edata['y'] = torch.zeros(edges)

        shift_index = (torch.sum(state[:,shift_feature_count:],1) == 0).nonzero(as_tuple=True)[0][0]

        if shift_index > 0:
            for i in range(shift_index):
                # get assignment data
                shift = i
                assigned_emp = (state[shift,shift_feature_count:] == 1).nonzero(as_tuple=True)[0]
                
                # get edge ids
                edges = bg.edges()[0]

                # index for shift-emp edge
                se_idx = (edges == shift).nonzero(as_tuple=True)[0][assigned_emp]

                # index for emp-shift edge
                emp_id = torch.unique(edges)[num_shifts:][assigned_emp]
                es_idx = (edges == emp_id).nonzero(as_tuple=True)[0][shift]
                
                #save data to edges
                bg.edges[[se_idx,es_idx]].data['y'] = torch.ones(2) # 2 because it's a bidirectional edge

        return bg, shift_index

    def forward(self, state, count_shifts, shift_features):
        graph, shift_index = self.grapher(state, shift_features)
        encoded_graph = self.encoder(graph, graph.ndata['x'],graph.edata['y'])
        self.probs = self.decoder(encoded_graph, shift_index, count_shifts)

        return self.probs
    
    def act(self, state, count_shifts, shift_features):
        probs = self.forward(state, count_shifts, shift_features)

        model = Categorical(probs)

        action = model.sample()
        return action.item(), model.log_prob(action)

    def guide(self, state, count_shifts, shift_features, method='max'):
        if method == 'max':
            probs = self.forward(state, count_shifts, shift_features)
            action = torch.argmax(probs)
        
        elif method == 'sample':
            probs = self.forward(state, count_shifts, shift_features)
            model = Categorical(probs)
            action = model.sample()

        return action.item()




