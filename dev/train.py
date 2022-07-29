import os
import pandas as pd
from collections import deque
from policy import *
from agent import reinforce, randomAgent
from datetime import datetime
from plot_utils import plot_scores, plot_learning_curve
from data_utils import problemLoader
import random
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

now = datetime.now().strftime("%m%d%H:%M")

# parameters
seeds= [10,21,65,100,999]

max_shifts = 14
n_episodes = 10000
print_every=100
reward_type = ['Step', 'Terminal','Step_Bonus']
problem_batch = problemLoader(max_shifts)
print(f"unique problems: {len(problem_batch)}")

for seed in seeds:
  print(seed)
  random.seed(seed)
  torch.manual_seed(seed)

  for rt in reward_type:
    print(rt)
    # model

    filename = f"{now}_seed{seed}_eps{n_episodes}_ms{max_shifts}_batchlen{len(problem_batch)}_rt{rt}"

    device = torch.device("cpu") #"cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(32, 32, 32)
    decoder = Decoder()
    policy = Policy(encoder, decoder).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=1e-3) # 1e-2
    agent = reinforce(policy, optimizer, reward_type=rt, max_t=1000,gamma=1)
    #agent = randomAgent()

    # experiments
    models_dir = f'models/{filename}'
    logdir = f'logs/{filename}'

    if not os.path.exists(models_dir):
      os.makedirs(models_dir)

    if not os.path.exists(logdir):
      os.makedirs(logdir)

    writer = SummaryWriter(log_dir=logdir)

    scores_deque = deque(maxlen=100)
    scores = []
    problog = []
    avg_scores = []

    for e in range(1, n_episodes):
      problem = random.choice(problem_batch)
      #print(problem)
      # problem_batch.pop(prob)
      problog.append(problem)

      reward, policy_loss = agent.run(problem, e, print_every)
      writer.add_scalar("Loss/train", policy_loss, e)
      writer.add_scalar("reward", reward, e)
      writer.add_scalar("avg_reward", np.mean(scores_deque), e)

      # Calculate total expected reward
      scores_deque.append(reward)
      scores.append(reward)
      avg_scores.append(np.mean(scores_deque))

      if e % print_every == 0:
          print('Episode {}\tAverage Score: {:.2f}'.format(e, np.mean(scores_deque)))
          
          torch.save(policy.state_dict(), f"{models_dir}/{e}.pth") # 'model_weights.pth')

    writer.flush()
    writer.close()       

    data = pd.DataFrame.from_dict({'avg_scores':avg_scores, 'scores':scores, 'prob':problog})
    data.to_csv(f"run_data/{filename}.csv",index=False)

