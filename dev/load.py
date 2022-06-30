from policy import *
from agent import reinforce, randomAgent


models_dir = f'models/'
logdir = f'logs/'

model_file = ''

# model
device = torch.device("cpu") #"cuda:0" if torch.cuda.is_available() else "cpu")
encoder = Encoder(32, 32, 32)
decoder = Decoder()
policy = Policy(encoder, decoder).to(device)

policy.load_state_dict(torch.load(f'{models_dir}{model_file}.pth'))

optimizer = optim.Adam(policy.parameters(), lr=1e-3) # 1e-2
agent = reinforce(policy, optimizer, reward_type, max_t=1000,gamma=1)

