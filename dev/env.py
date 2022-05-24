from sre_parse import State
from xmlrpc.client import Boolean
from gym import Env
from gym.spaces import Discrete, Box, Dict, MultiBinary
import numpy as np
import pandas as pd


class SchedulingEnv(Env):
    """A personnel scheduling environment for OpenAI gym"""

    def __init__(self, pool, schedule):
        schedule = pd.get_dummies(schedule,drop_first=True)
        self.shift_features = schedule.shape[1]
        for i in pd.get_dummies(pool).columns.to_list():
            schedule[i] = 0
        
        self.schedule = schedule
        self.state = self.schedule.to_numpy()
        self.count_workers = len(pool)
        self.count_shifts = len(schedule)
        self.shift_number = 0
        
        # action space: Employees we can assign to shifts
        self.action_space = Discrete(self.count_workers)
        # observation space: the latest state matrix
        self.observation_space = Box(low=0, high=1, shape=(self.state.shape[0], self.state.shape[1]),\
                                     dtype=np.float64)

 
    def step(self, action):
        """Step through the environment and implement action.

        :param action: the action returned by the policy; a employee to assign to a shift.
        :type action: int

        :returns:
            -self.state (:py:class:`numpy.ndarray`) - the state matrix updated following the action taken
            -reward (:py:class:`int`) - the reward follwing the action. No reward if done = False
            -done (:py:class:`boolean`) - flag to indicate whether the episode is complete
            -info (:py:class:`int`) - ?
        """
    
        # assign worker
        self.state[self.shift_number,(self.shift_features+action)] = 1

        # done
        all_shifts_assigned_check = self.state[:,self.shift_features:].sum() 
        if all_shifts_assigned_check < self.count_shifts:
            done = False
            self.shift_number += 1
        else:
            done = True
        
        # reward 
        # 0 - constraint violations
        # 1 - no constraint violations
 
        if done == False:
             reward = 0
        
        else:
            # check if an employee worked twice on the same day 
            # if so, b2b shift penalty should be applied
            b2b_penalty = 0
            b2b_penalty += 1 if self.state[:,self.shift_features][0:2].sum() == 0 or \
                                self.state[:,self.shift_features][0:2].sum() > 1 else 0
            b2b_penalty += 1 if self.state[:,self.shift_features][2:4].sum() == 0 or \
                                self.state[:,self.shift_features][2:4].sum() > 1  else 0
            
            if b2b_penalty > 0:
                reward = 0
            else:
                reward = 1

            print('summary:')
            print(self.state)
            print(reward)

        # info placeholder
        info = {}

        return self.state, reward, done, info
 
    def render(self):
        # graph viz?
        pass
    
    def reset(self):
        """Reset the environment to the starting state.

        :return: starting state matrix
        :rtype: numpy.array
        """
        self.shift_number = 0
        self.state = self.schedule.to_numpy()
        return self.state
