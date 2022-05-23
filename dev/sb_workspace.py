from env import SchedulingEnv
from stable_baselines3.common import env_checker
import pandas as pd


pool, schedule = pd.read_csv('dev/schedules/simple_pool.csv',dtype={'employee_id':'str'}), \
                 pd.read_csv('dev/schedules/simple_schedule.csv',dtype={'shift_id':'str'})

env = SchedulingEnv(pool, schedule)

env_checker.check_env(env=env,warn=True,skip_render_check=True)