import pandas as pd
from env import SchedulingEnv
from policy import *
from agent import reinforce, randomAgent
import torch.optim as optim
from datetime import datetime
from util import plot_scores, plot_learning_curve

now = datetime.now().strftime("%Y-%m-%d-%H:%M")

pool, schedule = pd.read_csv('dev/schedules/simple_pool.csv',dtype={'employee_id':'str'}), \
                 pd.read_csv('dev/schedules/simple_schedule.csv',dtype={'shift_id':'str'})

env = SchedulingEnv(pool, schedule)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

encoder = Encoder(4, 32, 32)
decoder = Decoder()
policy = Policy(env.state, encoder, decoder).to(device)

optimizer = optim.Adam(policy.parameters(), lr=1e-3) # 1e-2
agent = reinforce(policy, optimizer,max_t=1000,gamma=1)
n_episodes = 10000
scores = agent.run(env,n_episodes=n_episodes,print_every=1000)

#agent = randomAgent()
#scores = agent.run(env, n_episodes)


filename = f"Reinforce_SchedulingEnv_{now}_max_t=1000_lr=1e-3_episodes={n_episodes}" #f"Random_SchedulingEnv_{now}"  

x = [i+1 for i in range(n_episodes -1)]
#plot_scores(scores, filename)
plot_learning_curve(scores, x, filename)


# mlfow (or tensorboard?)
# hyperparameters:
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
#   std 