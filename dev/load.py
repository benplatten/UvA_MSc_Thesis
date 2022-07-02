from policy import *
from agent import reinforce, randomAgent
from data_utils import loadTestProblem, testProbList
import torch.optim as optim
import glob
import random


### DATA ###
## shifts
# easy = 3-8 
# medium = 9-14
# hard = 15-18

## ratio
# mixed
# below mean (easier)
# above mean (hard)

test_sets = ['shifts_easy_ratio_mixed','shifts_medium_ratio_mixed','shifts_hard_ratio_mixed']
shifts_easy_ratio_mixed = testProbList('shifts_easy_ratio_mixed')

test_set_lists = []


### MODELS 
policy_method = 'max' # 'sample'
models_dir = 'test_models/'
model_files = ['stepbonus_8.pth']
model_file = model_files[0]
#'063015:28_seed10_eps10000_ms8_batchlen416_rtStep_Bonus/5000.pth'


### SEEDS
seeds= [10, 21, 31, 65, 91]
#seed=seeds[0]

            # set up tensorboard?

# for seed in seeds:
#     random.seed(seed)
#     torch.manual_seed(seed)

#     for model in model_files:

#         for setlist in test_set_lists:

#             for i in setlist:

problems=[]
rewards=[]
for i in shifts_easy_ratio_mixed:

    schedule, pool = pd.read_csv(f'{i[0]}',dtype={'shift_id':'str'}), \
                pd.read_csv(f'{i[1]}',dtype={'employee_id':'str'})

    ## solver ##
    device = torch.device("cpu") #"cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(32, 32, 32)
    decoder = Decoder()
    policy = Policy(encoder, decoder).to(device)

    policy.load_state_dict(torch.load(f'{models_dir}{model_file}'))

    optimizer = optim.Adam(policy.parameters(), lr=1e-3) # 1e-2
    agent = reinforce(policy, optimizer, reward_type='Step_Bonus', max_t=1000,gamma=1)

    reward = agent.solve(pool, schedule,method=policy_method)
    rewards.append(reward)
    problem = str(i[0].split('_')[-1].split('.')[0] + i[1].split('_')[-1].split('.')[0])
    problems.append(problem)

blank_test_data  = pd.DataFrame(columns=['problem','reward', 'model', 'seed'])
blank_test_data.to_csv('test_data/test_data.csv',index=False)
data = {'set': 'shifts_easy_ratio_mixed', 'problem': problems, 'reward': rewards, 'model': model_file, 'seed':str(seed)}
test_data = pd.DataFrame.from_dict(data)
main = pd.read_csv('test_data/test_data.csv')
new_main = main.append(test_data, ignore_index=True)
new_main.to_csv('test_data/test_data.csv',index=False)


#logdir = f'logs/'






