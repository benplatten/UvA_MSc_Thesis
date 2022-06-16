import pandas as pd
from env import SchedulingEnv
from policy import *
from agent import reinforce, randomAgent
import torch.optim as optim
from datetime import datetime
from util import plot_scores, plot_learning_curve
from data_utils import problemLoader


max_shifts=5
problem_batch = problemLoader(max_shifts)
# list of tuples with csv ids

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
encoder = Encoder(env.shift_features, env.count_workers, 32, 32, 32)
decoder = Decoder(env.count_shifts)
policy = Policy(encoder, decoder).to(device)

optimizer = optim.Adam(policy.parameters(), lr=1e-3) # 1e-2


agent = reinforce(policy, optimizer,max_t=1000,gamma=1)
n_episodes = 10
scores = agent.run2(problem_batch,n_episodes=n_episodes,print_every=1)




