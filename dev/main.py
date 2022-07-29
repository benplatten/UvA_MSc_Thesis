import pandas as pd
from env import SchedulingEnv
from policy import *
from agent import reinforce, randomAgent
import torch.optim as optim
from datetime import datetime
from plot_utils import plot_scores, plot_learning_curve
from data_utils import problemLoader

start=datetime.now()
now = datetime.now().strftime("%m%d%H:%M")
p = "pool_0001"
s = "schedule_0001"

pool, schedule = pd.read_csv(f'scheduling_problems/pools/{p}.csv',dtype={'employee_id':'str'}), \
                 pd.read_csv(f'scheduling_problems/schedules/{s}.csv',dtype={'shift_id':'str'})

# print(pool)
# print(schedule)

schedule['shift_day_of_week'] = schedule['shift_day_of_week'].replace(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],[1, 2, 3, 4, 5])
schedule['shift_type'] = schedule['shift_type'].replace(['Morning', 'Evening'],[1, 2])


env = SchedulingEnv(pool, schedule, reward_type='Terminal')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

encoder = Encoder(env.shift_features, env.count_workers, 32, 32, 32)
decoder = Decoder(env.count_shifts)
policy = Policy(encoder, decoder).to(device)

# optimizer = optim.Adam(policy.parameters(), lr=1e-3) # 1e-2
# agent = reinforce(policy, optimizer,max_t=1000,gamma=1)
# n_episodes = 2
# scores = agent.run(env,n_episodes=n_episodes,print_every=1000)

# #agent = randomAgent()
# #scores = agent.run(env, n_episodes)


 
#print(env.shift_features)
#print(env.count_shifts)
#print(env.count_workers)
#print(env.state) 
#print(env.action_space.sample())