from policy import *
from agent import reinforce, randomAgent
from data_utils import loadTestProblem, testProbList, problemLoader
import torch.optim as optim
import glob
import random

import warnings
warnings.filterwarnings("ignore")

### test probs ###
#prob  = loadTestProblem(num_shifts=False)
#print(prob)
# prob = ('scheduling_problems/test_set/shifts_extrahard_ratio_above/pool_45.csv', 'scheduling_problems/test_set/shifts_extrahard_ratio_above/schedule_45.csv')

# pool, schedule = pd.read_csv(f'{prob[0]}',dtype={'employee_id':'str'}), \
#                 pd.read_csv(f'{prob[1]}',dtype={'shift_id':'str'})

### train probs ###
prob = ('0049', '0007')

s = prob[0]
p = prob[1]

pool, schedule = pd.read_csv(f'scheduling_problems/pools/pool_{p}.csv',dtype={'employee_id':'str'}), \
                pd.read_csv(f'scheduling_problems/schedules/schedule_{s}.csv',dtype={'shift_id':'str'})


# print(pool)
# print(schedule)

### MODELS 
policy_method = 'max' # 'sample'
models_dir = 'test_models/'
model_files =  ['step_91','terminal_91','stepbonus_8_best'] # ['stepbonus_8','step_8','terminal_8']
model_file = 'stepbonus_14_999'

reward_type = ['Step', 'Terminal','Step Bonus'] 

## random ##
#agent = randomAgent()
#agent.solve(pool, schedule)

rt = reward_type[2]

max_shifts = 14
seed= 101
random.seed(seed)
torch.manual_seed(seed)

# problem_batch = problemLoader(max_shifts)
# print(f"unique problems: {len(problem_batch)}")


#for prob in problem_batch:
print(prob)

#for rt in reward_type:
        
## solver ##
device = torch.device("cpu") #"cuda:0" if torch.cuda.is_available() else "cpu")
encoder = Encoder(32, 32, 32)
decoder = Decoder()
policy = Policy(encoder, decoder).to(device)

policy.load_state_dict(torch.load(f'{models_dir}{model_file}.pth'))

optimizer = optim.Adam(policy.parameters(), lr=1e-3) # 1e-2
agent = reinforce(policy, optimizer, reward_type=rt, max_t=1000,gamma=1)

for i in range(1):
    print_every=5

    reward = agent.solve(pool, schedule,method=policy_method)
    #reward, policy_loss = agent.run(prob,e=i,print_every=print_every)

    if i % print_every == 0:
        print(reward)

