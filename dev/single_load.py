from policy import *
from agent import reinforce, randomAgent
from data_utils import loadTestProblem
from data_utils import loadTestProblem, testProbList
import torch.optim as optim
import glob
import random

prob  = loadTestProblem(num_shifts=False)
print(prob)
pool, schedule = pd.read_csv(f'{prob[0]}',dtype={'employee_id':'str'}), \
                pd.read_csv(f'{prob[1]}',dtype={'shift_id':'str'})

print(pool)
print(schedule)


### MODELS 
policy_method = 'max' # 'sample'
models_dir = 'test_models/'
model_files = ['stepbonus_8','step_8','terminal_8']
model_file = model_files[0]


## random ##

#agent = randomAgent()
#agent.solve(pool, schedule)


## solver ##
device = torch.device("cpu") #"cuda:0" if torch.cuda.is_available() else "cpu")
encoder = Encoder(32, 32, 32)
decoder = Decoder()
policy = Policy(encoder, decoder).to(device)

policy.load_state_dict(torch.load(f'{models_dir}{model_file}.pth'))

optimizer = optim.Adam(policy.parameters(), lr=1e-3) # 1e-2
agent = reinforce(policy, optimizer, reward_type='Step_Bonus', max_t=1000,gamma=1)

agent.solve(pool, schedule,method=policy_method)


