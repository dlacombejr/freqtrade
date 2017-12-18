import os
import glob
import progressbar
import pandas as pd
from datetime import datetime

# define market
market = 'BCHBTC'

# define trade inteval
trade_interval = 1

# ensure directory exists
base_path = os.path.join(os.path.expanduser('~'), 'freqtrade', 'ml_dev', 'data')
market_path = os.path.join(base_path, 'hitbtc', market, str(trade_interval) + 'min')

# get all of the dataframes for the market
file_paths = glob.glob(os.path.join(market_path, '*.csv'))

# aggregate dataframes into a master
master = None
for file_index in range(len(file_paths)):

    df = pd.read_csv(file_paths[file_index])

    if file_index == 0:
        master = df
    else:
        master = pd.concat([master, df])

# sort by date
master = master.sort_values(by='date')

# reset indices
master = master.reset_index(drop=True)

# # get targets
# master['percent_change'] = 0.
# bar = progressbar.ProgressBar()
# entry_ind = 0
# subtract_index = 0
# look_ahead = 1
# for entry in bar(range(len(master) - look_ahead - 1)):
#
#     # get entry index
#     entry_ind = entry - subtract_index
#
#     # check that the two intervals are two minutes apart
#     # todo: check three ahead
#     if len(str(master['date'][entry_ind])) == 10:
#         date_string = "%Y-%m-%d"
#     else:
#         date_string = "%Y-%m-%d %H:%M:%S"
#     t0 = datetime.strptime(str(master['date'][entry_ind]), date_string)
#     if len(str(master['date'][entry_ind + look_ahead])) == 10:
#         date_string = "%Y-%m-%d"
#     else:
#         date_string = "%Y-%m-%d %H:%M:%S"
#     t1 = datetime.strptime(str(master['date'][entry_ind + look_ahead]), date_string)
#
#     # get difference in minutes
#     diff = (t1 - t0).total_seconds() / 60.
#
#     if diff == trade_interval:
#
#         # get price percentage change at t+1
#         percentage_change = (master['bid'][entry_ind + look_ahead + 1] -
#                              master['ask'][entry_ind + look_ahead]) / master['ask'][entry_ind + look_ahead]
#         master['percent_change'][entry_ind] = percentage_change
#
#     else:
#
#         # drop that entry
#         master = master.drop(master.index[entry_ind])
#         master = master.reset_index(drop=True)
#         subtract_index += 1
#
# print('{} Entries'.format(entry_ind))
#
# # delete last entry as we don't have next entry to get target for
# master = master.drop(master.index[len(master) - 3:])

# save master
master['date'] = pd.to_datetime(master.date, format="%Y-%m-%d %H:%M:%S")
master = master.sort_values(by='date')
if not os.path.isdir(os.path.join(market_path, 'master')):
    os.mkdir(os.path.join(market_path, 'master'))
master.to_csv(os.path.join(market_path, 'master', 'master.csv'), index=False)
