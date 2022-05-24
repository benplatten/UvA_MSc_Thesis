import pandas as pd
from env import SchedulingEnv
from policy import *
from agent import reinforce
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime

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
scores = agent.run(env,n_episodes=20)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(f'plots/Reinforce_SchedulingEnv_{now}.png')
#plt.show() 
