import numpy as np
import torch
from collections import deque
import pandas as pd
from env import SchedulingEnv
import random

class reinforce():
    def __init__(self, policy, optimizer,reward_type, max_t=1000, gamma=1.0):
        self.policy = policy
        self.optimizer = optimizer
        self.max_t = max_t
        self.gamma = gamma
        self.reward_type = reward_type

    def old_run(self, problem_batch, n_episodes=1000, print_every=100):
        scores_deque = deque(maxlen=100)
        scores = []
        problog = []

        for e in range(1, n_episodes):
            
            prob = random.choice(problem_batch)
            # problem_batch.pop(prob)
            problog.append(prob)

            s = prob[0]
            p = prob[1]

            pool, schedule = pd.read_csv(f'scheduling_problems/pools/pool_{p}.csv',dtype={'employee_id':'str'}), \
                            pd.read_csv(f'scheduling_problems/schedules/schedule_{s}.csv',dtype={'shift_id':'str'})

            schedule['shift_day_of_week'] = schedule['shift_day_of_week'].replace(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],[1, 2, 3, 4, 5])
            schedule['shift_type'] = schedule['shift_type'].replace(['Morning', 'Evening'],[1, 2])

            env = SchedulingEnv(pool, schedule,self.reward_type)

            #print(f"episode: {e}")
            saved_log_probs = []
            rewards = []
            state = env.reset()
            # Collect trajectory
            for t in range(self.max_t):
                # Sample the action from current policy
                action, log_prob = self.policy.act(state, env.count_shifts, env.shift_features)
                saved_log_probs.append(log_prob)
                state, reward, done, _ = env.step(action)
                rewards.append(reward)
                if done:
                    break
            # Calculate total expected reward
            scores_deque.append(sum(rewards))
            scores.append(sum(rewards))
            
            # Recalculate the total reward applying discounted factor
            discounts = [self.gamma ** i for i in range(len(rewards) + 1)]
            R = sum([a * b for a,b in zip(discounts, rewards)])
            
            # Calculate the loss 
            policy_loss = []
            for log_prob in saved_log_probs:
                # Note that we are using Gradient Ascent, not Descent. So we need to calculate it with negative rewards.
                policy_loss.append(-log_prob * R)
            # After that, we concatenate whole policy loss in 0th dimension
            policy_loss = torch.stack(policy_loss).sum()
            
            # Backpropagation
            self.optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            self.optimizer.step()
            
            if e % print_every == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(e, np.mean(scores_deque)))
                print(env.state)
  
        return scores, problog

    def run(self, problem, e, print_every):

            s = problem[0]
            p = problem[1]

            pool, schedule = pd.read_csv(f'scheduling_problems/pools/pool_{p}.csv',dtype={'employee_id':'str'}), \
                            pd.read_csv(f'scheduling_problems/schedules/schedule_{s}.csv',dtype={'shift_id':'str'})

            schedule['shift_day_of_week'] = schedule['shift_day_of_week'].replace(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],[1, 2, 3, 4, 5])
            schedule['shift_type'] = schedule['shift_type'].replace(['Morning', 'Evening'],[1, 2])

            env = SchedulingEnv(pool, schedule, self.reward_type)

            #print(f"episode: {e}")
            saved_log_probs = []
            rewards = []
            state = env.reset()
            # Collect trajectory
            for t in range(self.max_t):
                # Sample the action from current policy
                action, log_prob = self.policy.act(state, env.count_shifts, env.shift_features)
                saved_log_probs.append(log_prob)
                state, reward, done, _ = env.step(action)
                rewards.append(reward)
                if done:
                    break
            
            # Recalculate the total reward applying discounted factor
            discounts = [self.gamma ** i for i in range(len(rewards) + 1)]
            R = sum([a * b for a,b in zip(discounts, rewards)])
            
            # Calculate the loss 
            policy_loss = []
            for log_prob in saved_log_probs:
                # Note that we are using Gradient Ascent, not Descent. So we need to calculate it with negative rewards.
                policy_loss.append(-log_prob * R)
            # After that, we concatenate whole policy loss in 0th dimension
            policy_loss = torch.stack(policy_loss).sum()
            
            # Backpropagation
            self.optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            self.optimizer.step()

            if e % print_every == 0:
                print(env.state)
            
            return sum(rewards), policy_loss

    def solve(self, pool, schedule,method='max'):
            # toggle sample / greedy

            schedule['shift_day_of_week'] = schedule['shift_day_of_week'].replace(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],[1, 2, 3, 4, 5])
            schedule['shift_type'] = schedule['shift_type'].replace(['Morning', 'Evening'],[1, 2])

            env = SchedulingEnv(pool, schedule, self.reward_type)
            
            print("Initial state:")
            print(env.state)

            #saved_log_probs = []
            rewards = []
            state = env.reset()
            # Collect trajectory
            for t in range(self.max_t):
                # Sample the action from current policy
                action = self.policy.guide(state, env.count_shifts, env.shift_features, method=method)
                #saved_log_probs.append(log_prob)
                state, reward, done, _ = env.step(action)
                rewards.append(reward)
                if done:
                    break
            
            print("Terminal state:")
            print(env.state)
            print(env.cummulative_reward)



class randomAgent():
    def run(self, problem, e, print_every):  

            s = problem[0]
            p = problem[1]

            pool, schedule = pd.read_csv(f'scheduling_problems/pools/pool_{p}.csv',dtype={'employee_id':'str'}), \
                            pd.read_csv(f'scheduling_problems/schedules/schedule_{s}.csv',dtype={'shift_id':'str'})

            schedule['shift_day_of_week'] = schedule['shift_day_of_week'].replace(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],[1, 2, 3, 4, 5])
            schedule['shift_type'] = schedule['shift_type'].replace(['Morning', 'Evening'],[1, 2])

            env = SchedulingEnv(pool, schedule, reward_type='Step_Bonus')
            scores_deque = deque(maxlen=100)
            scores = []

            #state = env.reset()
            rewards = []
            done = False

            while not done:
                random_action = env.action_space.sample()
                obs_, reward, done, _ = env.step(random_action)
                rewards.append(reward)

                # Calculate total expected reward
                #scores_deque.append(sum(rewards))
                #scores.append(sum(rewards))
                
                if e % print_every == 0:
                    #print('Episode {}\tAverage Score: {:.2f}'.format(e, np.mean(scores_deque)))
                    print(env.state)
                    
            return sum(rewards), 0