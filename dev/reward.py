
# reward 
# 
# 0 - 1
# 0.6 1 constraint violation
# 0.4 2 constraint violations
# look at baseline options in SB 

""" if done == False:
        reward = 0

else:
    assignment_reward = self.count_shifts
    # check if an employee worked twice on the same day 
    # if so, b2b shift penalty should be applied
    b2b_penalty = 0
    b2b_penalty += 1 if self.state[:,self.shift_features][0:2].sum() == 0 or \
                        self.state[:,self.shift_features][0:2].sum() > 1 else 0
    b2b_penalty += 1 if self.state[:,self.shift_features][2:4].sum() == 0 or \
                        self.state[:,self.shift_features][2:4].sum() > 1  else 0
    reward = assignment_reward - (b2b_penalty)  """

import numpy as np
state = np.array([[0, 0, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0, 0]])

shift_features = 5

b2b_penalty = 0
b2b_penalty += 1 if state[:,shift_features][0:2].sum() == 0 or \
                        state[:,shift_features][0:2].sum() > 1 else 0
b2b_penalty += 1 if state[:,shift_features][2:4].sum() == 0 or \
                        state[:,shift_features][2:4].sum() > 1  else 0

print(b2b_penalty)

