import pandas as pd
from env import SchedulingEnv
from policy import *
from agent import reinforce, randomAgent
import torch.optim as optim
from datetime import datetime
from util import plot_scores, plot_learning_curve
#from data_utils import problemLoader


#num_shifts = 8
#num_emps = 4
#problem_batch = problemLoader(num_shifts,num_emps)
# list of tuples with csv ids

p = "pool_0001"
s = "schedule_0001"

pool, schedule = pd.read_csv(f'dev/scheduling_problems/pools/{p}.csv',dtype={'employee_id':'str'}), \
                 pd.read_csv(f'dev/scheduling_problems/schedules/{s}.csv',dtype={'shift_id':'str'})

env = SchedulingEnv(pool, schedule)

#print(env.state)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
encoder = Encoder(env.shift_features, env.count_workers, 32, 32, 32)
decoder = Decoder(env.count_shifts)
policy = Policy(encoder, decoder).to(device)

optimizer = optim.Adam(policy.parameters(), lr=1e-3) # 1e-2
agent = reinforce(policy, optimizer,max_t=1000,gamma=1)
n_episodes = 2
scores = agent.run(env,n_episodes=n_episodes,print_every=1000)


#print(bg.number_of_edges())
#print(bg.edata)
#print(bg.edges(form='all')[1])


#G.edges[1, 2].data['y'] = th.ones((1, 4))
#G.edata

