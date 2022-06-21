from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import pandas as pd
import joblib
from numpy.lib.stride_tricks import sliding_window_view


class SchedulingEnv(Env):
    """A personnel scheduling environment for OpenAI gym"""

    def __init__(self, pool, schedule, reward_type='Terminal'):
        sfEncodings = joblib.load('dev/shiftFeatureEncoding.joblib')
        shifts = pd.get_dummies(schedule[['shift_id']],drop_first=True)
        sfEncoded =  sfEncodings.transform(schedule[['shift_day_of_week','shift_type']])
        shift_features = pd.DataFrame(sfEncoded, columns=sfEncodings.get_feature_names_out())
        schedule = pd.merge(shifts, shift_features, left_index=True, right_index=True)

        self.shift_features = schedule.shape[1]
        # OR self.shift_features = shift_features.shape[1]

        for i in pd.get_dummies(pool).columns.to_list():
            schedule[i] = 0
        
        self.schedule = schedule
        self.state = self.schedule.to_numpy()

        # sf_index may not be needed here
        sf_start = len(schedule)-1
        sf_end = schedule.shape[1] - len(pool)
        self.sf_index = (sf_start,sf_end)

        self.count_workers = len(pool)
        self.count_shifts = len(schedule)
        self.shift_number = 0
        self.reward_type = reward_type
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
        

        #print(f"reward_step:{self.reward_step}")
        #print(f"shift_num:{self.shift_number}")
        #print(self.state)

        if self.reward_type == 'Terminal':
            if done == True:
                count_b2b_violation = self.evaluateSchedule()
                reward = 1 - (count_b2b_violation / self.count_shifts)
            
            elif done == False:
                reward = 0

        elif self.reward_type == 'Step':

            if self.reward_step == 0:
                reward = 0
            
            elif self.reward_step > 0:
                count_b2b_violation = self.evaluateStep()
                reward = 1 - count_b2b_violation
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

    def evaluateSchedule(self):
        """Check a completed schedule for constraint violations.
        For each employee, apply sliding window to compare successive shifts.
        If an employee is assigned to b2b shifts, look up the relevant features.
        If the shifts are on the same day, record a constraint violation
        If the shifts are on successive days, evening then morning, records a constraint  violation.
        Else, no constraint violation.

        :param state: state matrix
        :type state: numpy.array
        :return: count of constraint violations
        :rtype: int
        """
        count_b2b_violation = 0

        for i in range(self.count_workers):
            # using sliding window to compare successive shifts
            # checks = a list of pairwise binary shift assignment feature, from last - first
            checks = sliding_window_view(self.state[:,self.shift_features+i], 2)[::-1]
            #print(checks)
            # for each check
            for j,k in enumerate(checks):
                shift_id = abs(j - len(checks))-1
                # check for b2b shifts
                # 1 = assigned, 0 = not assigned
                if sum(k) > 1:
                    #print(f"employee:{i}, shift:{shift_id},{k}")
                    # get features for b2b shifts
                    shift_feats = self.state[shift_id:shift_id+2,self.count_shifts-1:self.shift_features]

                    # just the features for day of week
                    day_feats_1 = self.state[shift_id,self.count_shifts-1:self.shift_features-1]
                    day_feats_2 = self.state[shift_id+1,self.count_shifts-1:self.shift_features-1]
                    day1 = [np.where(day_feats_1==1)[0].item() + 2 if np.where(day_feats_1==1)[0].size != 0 else 1][0]
                    day2 = [np.where(day_feats_2==1)[0].item() + 2 if np.where(day_feats_2==1)[0].size != 0 else 1][0]
                    
                    # shifts are on the same day = violation
                    if day1 == day2:
                        count_b2b_violation += 1
                        print(f"shift:{shift_id+1},employee:{i}, constraint1 violated")
                    
                    # if shifts are on successive days, evening -> morning = violation
                    if day2 == day1+1:
                        # if shift 1 type = evening and shift 2 type = morning, record violation
                        count_b2b_violation += [1 if shift_feats[:,4][0] == 1 and shift_feats[:,4][1] == 0 else 0][0]
                        if [1 if shift_feats[:,4][0] == 1 and shift_feats[:,4][1] == 0 else 0][0] == 1:
                            print(f"shift:{shift_id+1},employee:{i}, constraint2 violated")


        return count_b2b_violation

    def evaluateStep(self):
        """Check the last 2 shift assignments for constraint violations.
        For each employee, compare successive shifts.
        If an employee is assigned to b2b shifts, look up the relevant features.
        If the shifts are on the same day, record a constraint violation
        If the shifts are on successive days, evening then morning, records a constraint  violation.
        Else, no constraint violation.

        :param state: state matrix
        :type state: numpy.array
        :return: count of constraint violations
        :rtype: int
        """
        count_b2b_violation = 0

        for i in range(self.count_workers):
            assignments = self.state[self.reward_step-1:self.reward_step+1,self.shift_features:][:,i]
            if sum(assignments) > 1:
                shift_feats = self.state[self.reward_step-1:self.reward_step+1,self.count_shifts-1:self.shift_features]

                # just the features for day of week
                day_feats_1 = self.state[self.reward_step-1,self.count_shifts-1:self.shift_features-1]
                day_feats_2 = self.state[self.reward_step,self.count_shifts-1:self.shift_features-1]
                day1 = [np.where(day_feats_1==1)[0].item() + 2 if np.where(day_feats_1==1)[0].size != 0 else 1][0]
                day2 = [np.where(day_feats_2==1)[0].item() + 2 if np.where(day_feats_2==1)[0].size != 0 else 1][0]
                
                # shifts are on the same day = violation
                if day1 == day2:
                    count_b2b_violation += 1
                    print(f"shift:{self.reward_step},employee:{i}, constraint1 violated")
                
                # if shifts are on successive days, evening -> morning = violation
                if day2 == day1+1:
                    # if shift 1 type = evening and shift 2 type = morning, record violation
                    count_b2b_violation += [1 if shift_feats[:,4][0] == 1 and shift_feats[:,4][1] == 0 else 0][0]
                    if [1 if shift_feats[:,4][0] == 1 and shift_feats[:,4][1] == 0 else 0][0] == 1:
                        print(f"shift:{self.reward_step},employee:{i}, constraint2 violated")

        return count_b2b_violation
