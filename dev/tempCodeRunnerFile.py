pool, schedule = pd.read_csv(f'dev/pools/{p}.csv',dtype={'employee_id':'str'}), \
                 pd.read_csv(f'dev/schedules/{s}.csv',dtype={'shift_id':'str'})
