from gym import Env
from gym.spaces import Discrete, Box
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
        self.reward_step = 0
        self.cum_reward = 0
        
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
        
        # reward function 1
        # 0 - constraint violations
        # 1 - no constraint violations

        # reward function 2
        # 1-(count_b2b / num shifts) - constraint violations
        # 1 - no constraint violations

        # reward function 3
        # 1/num shifts for each acceptable assignment

        # reward function 4
        # end episode if b2b contraint broken 

        # reward function 5
        # 0 reward if b2b contraint broken 

        #


        #print(f"reward_step:{self.reward_step}")
        #print(f"shift_num:{self.shift_number}")
        #print(self.state)
 
        if self.reward_step == 0:
            reward = 0
            #print(f"reward:{reward}")
        
        else:
            step_b2bs = 0 
            for i in range(self.state[self.reward_step-1:self.reward_step+1,self.shift_features:].shape[1]):
                step = self.state[self.reward_step-1:self.reward_step+1,self.shift_features:][:,i]
                #print(step)
                step_b2bs += self.check_b2b(step)

            if step_b2bs == 0:
                reward = (1/(self.count_shifts-1)) 

            else: 
                #done = True
                #print(f"Episode ended on shift: {self.shift_number}")
                reward = self.cum_reward * -1  
            #print(f"reward:{reward}")

        self.reward_step += 1
        self.cum_reward += reward

        #if done:
            #print(f"episode reward: {self.cum_reward}")

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
        self.reward_step = 0
        self.cum_reward = 0
        self.state = self.schedule.to_numpy()
        return self.state

    def check_b2b(self, worker_array):
        countb2b=0
        count=0
        for i in range(len(worker_array)):
            if worker_array[i]==1:
                count=count+1
                if count > 1:
                    countb2b += 1                       
            else:
                count=0

        return countb2b
