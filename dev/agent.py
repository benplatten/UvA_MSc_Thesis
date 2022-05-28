import numpy as np
import torch
from collections import deque

class reinforce():
    def __init__(self, policy, optimizer, max_t=1000, gamma=1.0):
        self.policy = policy
        self.optimizer = optimizer
        self.max_t = max_t
        self.gamma = gamma

    def run(self, env, n_episodes=1000, print_every=100):
        scores_deque = deque(maxlen=100)
        scores = []
        for e in range(1, n_episodes):
            #print(f"episode: {e}")
            saved_log_probs = []
            rewards = []
            state = env.reset()
            # Collect trajectory
            for t in range(self.max_t):
                # Sample the action from current policy
                action, log_prob = self.policy.act(state)
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
            #if np.mean(scores_deque) >= 0.8:
            #    print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(e - 100, np.mean(scores_deque)))
            #    break
  
        return scores


class randomAgent():
    def run(self, env, n_episodes=1000, print_every=100):  
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