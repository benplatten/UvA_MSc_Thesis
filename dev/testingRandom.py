from env import SchedulingEnv
import numpy as np
import pandas as pd
from collections import deque

pool, schedule = pd.read_csv('dev/schedules/simple_pool.csv',dtype={'employee_id':'str'}), \
                 pd.read_csv('dev/schedules/simple_schedule.csv',dtype={'shift_id':'str'})

env = SchedulingEnv(pool, schedule)


def run(env, n_episodes=1000, print_every=100):  # self, 
        scores_deque = deque(maxlen=100)
        scores = []

        for e in range(1, n_episodes):
            obs = env.reset()
            rewards = []
            done = False

            while not done:
                random_action = env.action_space.sample()
                obs_, reward, done, _ = env.step(random_action)
                rewards.append(reward)

            # Calculate total expected reward
            scores_deque.append(sum(rewards))
            scores.append(sum(rewards))
            
            if e % print_every == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(e, np.mean(scores_deque)))
                print(env.state)
            if np.mean(scores_deque) >= 0.8:
                print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(e - 100, np.mean(scores_deque)))
                break
                
        return scores

scores = run(env)