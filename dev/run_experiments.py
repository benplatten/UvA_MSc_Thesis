import pandas as pd
from env import SchedulingEnv
from policy import *
from agent import reinforce, randomAgent
import torch.optim as optim
from datetime import datetime
from plot_utils import plot_scores, plot_learning_curve
from data_utils import problemLoader
import random

import warnings
warnings.filterwarnings("ignore")

seeds= [10, 21, 31, 65, 91]
seed=seeds[3]
random.seed(seed)

now = datetime.now().strftime("%m%d%H:%M")
max_shifts = 8
n_episodes = 10000
reward_type = 'Step' # 'Terminal' 'Step'

problem_batch = problemLoader(max_shifts)

device = torch.device("cpu") #"cuda:0" if torch.cuda.is_available() else "cpu")
encoder = Encoder(32, 32, 32)
decoder = Decoder()
policy = Policy(encoder, decoder).to(device)

optimizer = optim.Adam(policy.parameters(), lr=1e-3) # 1e-2
agent = reinforce(policy, optimizer, reward_type, max_t=1000,gamma=1)

scores, problog = agent.run(problem_batch,n_episodes=n_episodes,print_every=100)

### viz
x = [i+1 for i in range(n_episodes -1)]
filename = f"{now}_max_shifts={max_shifts}_reward_type={reward_type}_episodes={n_episodes}_seed={seed}" #f"Random_SchedulingEnv_{now}"  
#plot_scores(scores, filename)
plot_learning_curve(scores, x, filename)

data = pd.DataFrame.from_dict({'scores':scores, 'prob':problog})
data.to_csv(f"dev/run_data/{filename}.csv",index=False)

