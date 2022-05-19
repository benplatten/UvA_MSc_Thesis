import pandas as pd
from env import SchedulingEnv


pool, schedule = pd.read_csv('schedules/simple_pool.csv',dtype={'employee_id':'str'}), \
                 pd.read_csv('schedules/simple_schedule.csv',dtype={'shift_id':'str'})

env = SchedulingEnv(pool, schedule)