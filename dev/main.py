import pandas as pd
from env import SchedulingEnv
from policy import *


pool, schedule = pd.read_csv('dev/schedules/simple_pool.csv',dtype={'employee_id':'str'}), \
                 pd.read_csv('dev/schedules/simple_schedule.csv',dtype={'shift_id':'str'})

env = SchedulingEnv(pool, schedule)

print(env.state)

encoder = Encoder(4, 32, 32)
decoder = Decoder()
policy = Policy(env.state, encoder, decoder)

print(policy.state)
print(policy.forward())

