import pandas as pd
import numpy as np
import glob
import random

def updateDataIndex():
    """Update index of scheduling data by iterating through schedule and pool directories.

    :return: Index of scheduling data.
    :rtype: pandas.DataFrame
    """
    scheds = glob.glob("/Users/benplatten/workspace/UvA_Thesis/UvA_MSc_Thesis/dev/scheduling_problems/schedules/*.csv")
    if len(scheds) == 0:
        print('problem loading data...')
    pools = glob.glob("/Users/benplatten/workspace/UvA_Thesis/UvA_MSc_Thesis/dev/scheduling_problems/pools/*.csv")
    s_len = []
    Schedule = []
    for sched in sorted(scheds):
        s_len.append(len(pd.read_csv(sched)))
        Schedule.append(sched.split('/')[9].split('_')[1].split('.')[0])
    p_len = []
    Pool = []
    for pool in sorted(pools):
        p_len.append(len(pd.read_csv(pool)))
        Pool.append(pool.split('/')[9].split('_')[1].split('.')[0])

    Pool += ['nan'] * (len(Schedule) - len(Pool))
    p_len += ['nan'] * (len(s_len) - len(p_len))

    data = {'Schedule':Schedule,'shifts':s_len,'Pool':Pool,'employees':p_len}

    di = pd.DataFrame(data)
    
    return di

def problemIndex():
    di = pd.read_csv('scheduling_problems/data_index.csv',dtype=str) 
    #di = updateDataIndex()
    di['shifts'] = di['shifts'].astype(int) 
    
    pi = pd.DataFrame(columns=['Schedule', 'Pool'])

    scheds = di.Schedule.to_list()
    pools = di.Pool.to_list()
    pools = [x for x in pools if str(x) != 'nan']

    for sched in scheds:
        for pool in pools:
            pi.loc[0 if pd.isnull(pi.index.max()) else pi.index.max() + 1] = [sched] + [pool]

    pi = pd.merge(pi, di[['Schedule', 'shifts']], on ='Schedule', how ='left')
    pi = pd.merge(pi, di[['Pool', 'employees']], on ='Pool', how ='left')
    pi['employees'] = pi['employees'].astype(int)
    pi['Nodes'] = pi['shifts'] + pi['employees']
    pi['Ratio'] = pi['shifts'] / pi['employees']

    print(f"{len(pi)} possible problems.")
    
    return pi

def matrify(pool, schedule):
    schedule = pd.get_dummies(schedule,drop_first=True)
    #shift_features = schedule.shape[1]
    for i in pd.get_dummies(pool).columns.to_list():
        schedule[i] = 0

    return schedule.to_numpy()

def problemValidation():
    scheds = glob.glob("scheduling_problems/schedules/*.csv")
    pools = glob.glob("scheduling_problems/pools/*.csv")

    stateDict = {}
    for sched in scheds:
        for pool in pools:
            id = sched.split('/')[2].split('_')[1].split('.')[0]+pool.split('/')[2].split('_')[1].split('.')[0]
            s = pd.read_csv(sched,dtype={'shift_id':'str'})
            p = pd.read_csv(pool,dtype={'employee_id':'str'})
            stateDict[id] = matrify(p, s)

    dupes=[]
    for key, value in stateDict.items(): 
        for k, v in stateDict.items(): 
            if key != k:
                if np.array_equal(value, v):
                    dupes.append((key,k))

    print(f"{len(dupes)} duplicate problems found in data set.")

    return dupes

def problemLoader(max_shifts):
    pi = problemIndex()
    #pi.head()

    selectedProbs = pi[pi['shifts'] <=max_shifts]
    #selectedProbs = pi[(pi['shifts'] == num_shifts) & (pi['employees'] == num_emps)] 
    selectedProbs

    glob_list = []
    for i in range(len(selectedProbs)):
        s = selectedProbs['Schedule'].iloc[i]
        p = selectedProbs['Pool'].iloc[i]
        glob_list.append((s,p))

    return glob_list

def randomSchedule(n=1, min_shifts=2,max_shifts=16, max_shifts_per_day=4):
    for i in range(n):
        # shift_id
        n = random.randint(min_shifts, max_shifts)
        #print(n)
        shift_id = list(range(0, n))

        # shift_day_of_week
        shift_day_of_week = []
        days = ['Monday','Tuesday','Wednesday','Thursday','Friday']

        for i in range(len(shift_id)):
            choice = random.choice(days)
            if shift_day_of_week.count(choice) < max_shifts_per_day + 1:
                shift_day_of_week.append(choice)

        shift_day_of_week = sorted(shift_day_of_week, key=days.index)

        # shift_type
        shift_type = []
        shift_types = ['Morning','Evening']

        dayset = set(shift_day_of_week)
        sortedDayset = sorted(list(dayset), key=days.index)

        for i in sortedDayset:
            temp = []
            for j in range(shift_day_of_week.count(i)):
                temp.append(random.choice(shift_types))
            temp = sorted(temp, key=shift_types.index)
            for k in temp:
                shift_type.append(k)

        scheduleDic = {'shift_id':shift_id,'shift_day_of_week':shift_day_of_week,'shift_type':shift_type}

        schedule = pd.DataFrame(scheduleDic)

        scheds = sorted(glob.glob("scheduling_problems/schedules/*.csv"))
        id = int(scheds[-1].split('/')[2].split('_')[1].split('.')[0]) + 1

        schedule.to_csv(f'scheduling_problems/schedules/schedule_0{id}.csv',index=False)

        print(f"schedule_00{id} saved.")

def empGen(num_shifts,num_emps=False):
    lower = round(num_shifts / 2.8)
    upper = round(num_shifts / 1.2)
    employee_id = []
    e = random.randint(lower, upper)
    if num_emps:
        for i in range(num_emps):
            employee_id.append(''.join(np.random.randint(9,size=(6)).astype('str')))
    else:
        for i in range(e):
            employee_id.append(''.join(np.random.randint(9,size=(6)).astype('str')))
    
    return employee_id

def randomProblem(min_shifts=2,max_shifts=15, max_shifts_per_day=4,num_emps=False):
    
    # shift_id
    n = random.randint(min_shifts, max_shifts)
    shift_id = list(range(0, n))
    
    # shift_day_of_week
    shift_day_of_week = []
    days = ['Monday','Tuesday','Wednesday','Thursday','Friday']

    for i in range(len(shift_id)):
        choice = random.choice(days)
        if shift_day_of_week.count(choice) < max_shifts_per_day + 1:
            shift_day_of_week.append(choice)

    shift_day_of_week = sorted(shift_day_of_week, key=days.index)
    
    # shift_type
    shift_type = []
    shift_types = ['Morning','Evening']

    dayset = set(shift_day_of_week)
    sortedDayset = sorted(list(dayset), key=days.index)

    for i in sortedDayset:
        temp = []
        for j in range(shift_day_of_week.count(i)):
            temp.append(random.choice(shift_types))
        temp = sorted(temp, key=shift_types.index)
        for k in temp:
            shift_type.append(k)
            
    # employee_id
    while True:
        employee_id = empGen(len(shift_id),num_emps)
        if len(shift_id) / len(employee_id) <= 5:
            break
        
    scheduleDic = {'shift_id':shift_id,'shift_day_of_week':shift_day_of_week,'shift_type':shift_type}
    
    schedule = pd.DataFrame(scheduleDic)
    pool = pd.DataFrame({'employee_id':employee_id})
    
    return schedule, pool
        
def buildTestSet(n,min_shifts,max_shifts,num_emps=False):
    i = 0
    while i < n:
        try:
            schedule, pool = randomProblem(min_shifts=min_shifts,max_shifts=max_shifts,num_emps=num_emps)

            tstst = sorted(glob.glob("scheduling_problems/test_set/*.csv"))

            if tstst:
                id = int(tstst[-1].split('/')[2].split('_')[1].split('.')[0]) + 1
        
            else:
                id = 1

            ratio = len(schedule) / len(pool)

            if ratio > 1 and ratio < 3:
                i += 1
                schedule, pool = schedule.to_csv(f'scheduling_problems/test_set/schedule_{str(id).zfill(2)}.csv',index=False), \
                            pool.to_csv(f'scheduling_problems/test_set/pool_{str(id).zfill(2)}.csv',index=False) 
        
            else:
                continue
        
        except ValueError as verror:
            print(verror)
            break

        except NameError as nerror:
            print(nerror)
            break

        except:
            continue

def loadTestProblem(num_shifts=False):
    tstst = sorted(glob.glob("scheduling_problems/test_set/*.csv"))

    try:
        if num_shifts:
            shfts = 0
            while not shfts == num_shifts:
                n = random.randint(1, (len(tstst) / 2))
                p = tstst[n-1]
                s = tstst[n+(int(len(tstst) / 2)-1)]

                schedule = pd.read_csv(f'{tstst[n+(int(len(tstst) / 2)-1)]}',dtype={'shift_id':'str'})
                if len(schedule) == num_shifts:
                    shfts = num_shifts

        else:
            n = random.randint(1, (len(tstst) / 2))
            p = tstst[n-1]
            s = tstst[n+(int(len(tstst) / 2)-1)]

        return (p,s)
    except:
        print(f"No problem with {num_shifts} shifts.")

############

# //// training data \\\\

#randomSchedule(n=5, min_shifts=14,max_shifts=14, max_shifts_per_day=4)

#di = updateDataIndex()
#di.to_csv('scheduling_problems/data_index.csv',index=False)

#pi = problemIndex()
#pi.to_csv('scheduling_problems/problem_index.csv', index=False)

#dupes = problemValidation()
#df = pd.DataFrame(dupes)
#df.to_csv('scheduling_problems/duplicate_problems.csv', index=False)

# //// test data \\\\

buildTestSet(n=2,min_shifts=3,max_shifts=8)

# loadTestProblem(num_shifts=5)