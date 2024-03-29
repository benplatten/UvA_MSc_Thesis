from policy import *
from agent import reinforce, randomAgent
from data_utils import testProbList
import torch.optim as optim
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

test_sets = ['shifts_easy_ratio_mixed','shifts_medium_ratio_mixed','shifts_hard_ratio_mixed','shifts_extrahard_ratio_mixed',
            'shifts_easy_ratio_above','shifts_medium_ratio_above','shifts_hard_ratio_above','shifts_extrahard_ratio_above',
            'shifts_xxhard_ratio_mixed','shifts_xxhard_ratio_above'] #,'shifts_xxxhard_ratio_above']

shifts_easy_ratio_mixed = testProbList('shifts_easy_ratio_mixed')
shifts_medium_ratio_mixed = testProbList('shifts_medium_ratio_mixed')
shifts_hard_ratio_mixed = testProbList('shifts_hard_ratio_mixed')
shifts_extrahard_ratio_mixed = testProbList('shifts_extrahard_ratio_mixed')
shifts_xxhard_ratio_mixed = testProbList('shifts_xxhard_ratio_mixed')
shifts_easy_ratio_above = testProbList('shifts_easy_ratio_above')
shifts_medium_ratio_above = testProbList('shifts_medium_ratio_above')
shifts_hard_ratio_above = testProbList('shifts_hard_ratio_above')
shifts_extrahard_ratio_above = testProbList('shifts_extrahard_ratio_above')
shifts_xxhard_ratio_above = testProbList('shifts_xxhard_ratio_above')
#shifts_xxxhard_ratio_above = testProbList('shifts_xxxhard_ratio_above')

#test_sets = ['shifts_xxxhard_ratio_above']

for i in test_sets:
     print(f"{i}: {len(eval(i))}")


### MODELS 
policy_method = 'max' # 'sample'
models_dir = 'test_models/'
#model_files =['stepbonus_8','step_8','terminal_8', 'random_0'] # ['terminal_8'] #

model_files =['step_8_10', 'terminal_8_10', 'stepbonus_8_10', 'step_8_21','terminal_8_21', 'stepbonus_8_21', 
              'step_8_65','terminal_8_65','stepbonus_8_65','step_8_100','terminal_8_100','stepbonus_8_100',
              'step_8_999','terminal_8_999','stepbonus_8_999','random_0_0']

# model_files =[ 'step_14_10', 'terminal_14_10', 'stepbonus_14_10', 'step_14_21','terminal_14_21', 'stepbonus_14_21', 
#               'step_14_65','terminal_14_65','stepbonus_14_65','step_14_100','terminal_14_100','stepbonus_14_100',
#               'step_14_999','terminal_14_999','stepbonus_14_999']

model_files =['random_0_0']

### SEEDS
seeds= [10,21,65,100,999]


#///////////////////
for seed in seeds:
    random.seed(seed)
    torch.manual_seed(seed)
    print(f"seed: {seed}")

    for model in model_files:
            print(f"model: {model}")
            if model == 'random_0_0':
                agent = randomAgent()
                
            else:
                ## solver ##
                device = torch.device("cpu") #"cuda:0" if torch.cuda.is_available() else "cpu")
                encoder = Encoder(32, 32, 32)
                decoder = Decoder()
                policy = Policy(encoder, decoder).to(device)

                policy.load_state_dict(torch.load(f'{models_dir}{model}.pth'))


                optimizer = optim.Adam(policy.parameters(), lr=1e-3) # 1e-2
                agent = reinforce(policy, optimizer, reward_type='Step_Bonus', max_t=1000,gamma=1)
                
            for setlist in test_sets:
                print(setlist)

                problems=[]
                rewards=[]
                
                for i in eval(setlist):
                        problem = str(i[0].split('_')[-1].split('.')[0] + i[1].split('_')[-1].split('.')[0])
                        problems.append(problem)

                        schedule, pool = pd.read_csv(f'{i[0]}',dtype={'shift_id':'str'}), \
                                            pd.read_csv(f'{i[1]}',dtype={'employee_id':'str'})

                        if model == 'random_0':
                            reward = agent.solve(pool, schedule)
                            rewards.append(reward)

                        else:
                            reward = agent.solve(pool, schedule,method=policy_method)
                            rewards.append(reward)


                #blank_test_data  = pd.DataFrame(columns=['problem','reward', 'model', 'seed'])
                #blank_test_data.to_csv('test_data/test_data.csv',index=False)
                data = {'set': setlist, 'problem': problems, 'reward': rewards, 'model': model.rsplit('_', 1)[0], 'seed':seed} #model.split('_')[2]}
                test_data = pd.DataFrame.from_dict(data)
                main = pd.read_csv('test_data/test_data.csv')
                new_main = main.append(test_data, ignore_index=True)
                new_main.to_csv('test_data/test_data.csv',index=False)

