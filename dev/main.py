import pandas as pd
from env import SchedulingEnv
from policy import *
from agent import reinforce, randomAgent
import torch.optim as optim
from datetime import datetime
from util import plot_scores, plot_learning_curve

start=datetime.now()
now = datetime.now().strftime("%Y%m%d%H:%M")
p = "pool_0001"
s = "schedule_0001"

pool, schedule = pd.read_csv(f'dev/pools/{p}.csv',dtype={'employee_id':'str'}), \
                 pd.read_csv(f'dev/schedules/{s}.csv',dtype={'shift_id':'str'})

env = SchedulingEnv(pool, schedule)

#print(env.shift_features)
#print(env.count_shifts)
#print(env.count_workers)
print(env.state) 
#print(env.action_space.sample())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

encoder = Encoder(env.shift_features, env.count_workers, 32, 32, 32)
decoder = Decoder(env.count_shifts)
policy = Policy(encoder, decoder).to(device)

optimizer = optim.Adam(policy.parameters(), lr=1e-3) # 1e-2
agent = reinforce(policy, optimizer,max_t=1000,gamma=1)
n_episodes = 2
scores = agent.run(env,n_episodes=n_episodes,print_every=1000)

#agent = randomAgent()
#scores = agent.run(env, n_episodes)

filename = f"rf5_{now}_max_t=1000_lr=1e-3_episodes={n_episodes}_{p}_{s}" #f"Random_SchedulingEnv_{now}"  

x = [i+1 for i in range(n_episodes -1)]
#plot_scores(scores, filename)
plot_learning_curve(scores, x, filename)

print(f"time to complete: {datetime.now() - start}")

 
# Tracking
# mlfow or tensorboard?
#
# Hyperparameters:
#   max_t
#   lr
#   n_episodes
#   gamma
#   deque maxlen?
#   batch size
#   samples
# 
# Scores:
#   average
#   max
#   count of max?
#   min
#   std  """