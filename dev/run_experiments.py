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

p = "pool_0020"
s = "schedule_0025"

pool, schedule = pd.read_csv(f'dev/scheduling_problems/pools/{p}.csv',dtype={'employee_id':'str'}), \
                 pd.read_csv(f'dev/scheduling_problems/schedules/{s}.csv',dtype={'shift_id':'str'})

env = SchedulingEnv(pool, schedule)

print(env.state)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
encoder = Encoder(env.shift_features, env.count_workers, 32, 32, 32)
decoder = Decoder(env.count_shifts)
policy = Policy(encoder, decoder).to(device)

policy.grapher(env.state)




