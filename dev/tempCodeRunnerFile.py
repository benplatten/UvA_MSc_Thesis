p = "pool_0001"
s = "schedule_0001"

pool, schedule = pd.read_csv(f'dev/scheduling_problems/pools/{p}.csv',dtype={'employee_id':'str'}), \
                 pd.read_csv(f'dev/scheduling_problems/schedules/{s}.csv',dtype={'shift_id':'str'})
