import pandas as pd
from env import SchedulingEnv
from policy import *
from agent import reinforce
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
from util import plot_scores, plot_learning_curve

now = datetime.now().strftime("%Y-%m-%d-%H:%M")

pool, schedule = pd.read_csv('schedules/simple_pool.csv',dtype={'employee_id':'str'}), \
                 pd.read_csv('schedules/simple_schedule.csv',dtype={'shift_id':'str'})

env = SchedulingEnv(pool, schedule)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

encoder = Encoder(4, 32, 32)
decoder = Decoder()
policy = Policy(env.state, encoder, decoder).to(device)

optimizer = optim.Adam(policy.parameters(), lr=1e-2)
agent = reinforce(policy, optimizer,max_t=5)
n_episodes = 1000
scores = agent.run(env,n_episodes=n_episodes,print_every=10)


filename = f"Reinforce_SchedulingEnv_{now}"

x = [i+1 for i in range(n_episodes -1)]
#plot_scores(scores, filename)
plot_learning_curve(scores, x, filename)

