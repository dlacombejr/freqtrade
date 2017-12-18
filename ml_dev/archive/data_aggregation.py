import os
import glob
import pandas as pd
from datetime import datetime

# define market
market = 'BTC_ETH'

# ensure directory exists
base_path = os.path.join(os.path.expanduser('~'), 'freqtrade', 'ml_dev', 'data')
market_path = os.path.join(base_path, 'markets', market)

# get all of the dataframes for the market
file_paths = glob.glob(os.path.join(market_path, '*.csv'))

# aggregate dataframes into a master
master = None
for file_index in range(len(file_paths)):

    print(file_paths[file_index])
    df = pd.read_csv(file_paths[file_index])

    if file_index == 0:
        master = df
    else:
        master = pd.concat([master, df])

# save master
master['date'] =pd.to_datetime(master.date, format="%Y-%m-%d %H:%M:%S")
master = master.sort_values(by='date')
master.to_csv(os.path.join(base_path, 'master.csv'), index=False)

# load back in
master = pd.read_csv(os.path.join(base_path, 'master.csv'))

# get targets
master['percent_change'] = 0.
# print(master)
for entry in range(len(master) - 1):

    print(entry)

    # check that the two intervals are two minutes apart
    # print(str(master['date'][entry]))
    t0 = datetime.strptime(str(master['date'][entry]), "%Y-%m-%d %H:%M:%S")
    t1 = datetime.strptime(str(master['date'][entry + 1]), "%Y-%m-%d %H:%M:%S")
    # print(t0)
    # print(t1)
    diff = (t1 - t0).total_seconds() / 60.

    print(diff)

    if diff == 2.:

        # get price percentage change at t+1
        percentage_change = (master['close'][entry + 1] - master['close'][entry]) / master['close'][entry]
        print(percentage_change)
        master['percent_change'][entry] = percentage_change

    else:

        # drop that entry
        master.drop(master.index[entry])

# delete last entry as we don't have next entry to get target for
master.drop(master.index[len(master)-1])

# save master
master['date'] =pd.to_datetime(master.date, format="%Y-%m-%d %H:%M:%S")
master = master.sort_values(by='date')
master.to_csv(os.path.join(base_path, 'master.csv'), index=False)
