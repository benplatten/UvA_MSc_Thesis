from policy import *
from agent import reinforce, randomAgent
from data_utils import loadTestProblem
import torch.optim as optim

prob  = loadTestProblem(num_shifts=False)
pool, schedule = pd.read_csv(f'{prob[0]}',dtype={'employee_id':'str'}), \
                pd.read_csv(f'{prob[1]}',dtype={'shift_id':'str'})

print(pool)
print(schedule)


models_dir = f'models/'
logdir = f'logs/'

model_file = '063015:28_seed10_eps10000_ms8_batchlen416_rtStep_Bonus/5000.pth'

# parameters
policy_method = 'max' # 'sample'
problem_type = 'easy' # 'easy' 'hard' 'mixed' 

# model
device = torch.device("cpu") #"cuda:0" if torch.cuda.is_available() else "cpu")
encoder = Encoder(32, 32, 32)
decoder = Decoder()
policy = Policy(encoder, decoder).to(device)

policy.load_state_dict(torch.load(f'{models_dir}{model_file}'))

optimizer = optim.Adam(policy.parameters(), lr=1e-3) # 1e-2
agent = reinforce(policy, optimizer, reward_type='Step_Bonus', max_t=1000,gamma=1)

agent.solve(pool, schedule,method=policy_method)


