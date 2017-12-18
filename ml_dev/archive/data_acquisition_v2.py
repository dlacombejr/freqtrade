import os
import json
import time
import datetime
import pandas as pd
from freqtrade import exchange
from freqtrade.analyze import analyze_ticker


# define market and time interval
market = 'BTC_ETH'
time_interval = 5

# ensure directory exists
base_path = os.path.join(os.path.expanduser('~'), 'freqtrade')
data_path = os.path.join(base_path, 'ml_dev', 'data', 'markets')
market_path = os.path.join(data_path, market)
if not os.path.isdir(market_path):
	os.mkdir(market_path)

# get configuration
with open(os.path.join(base_path, 'config.json')) as file:
	_CONF = json.load(file)

# initialize the exchange
exchange.init(_CONF)

# get ticker hisroty
df = analyze_ticker(pair=market, tick_interval=time_interval)

# remove first 100 rows to allow ema100
df = df[100:]

df.to_csv('test.csv', index=False)

master = pd.read_csv('test.csv')

# get targets
master['percent_change'] = 0.
print(master.shape)
for entry in range(len(master) - 1):

    print(entry)

    # get price percentage change at t+1
    percentage_change = (master['close'][entry + 1] - master['close'][entry]) / master['close'][entry]
    print(percentage_change)
    master['percent_change'][entry] = percentage_change

# delete last entry as we don't have next entry to get target for
master.drop(master.index[len(master)-1])

# save master
master['date'] =pd.to_datetime(master.date, format="%Y-%m-%d %H:%M:%S")
master = master.sort_values(by='date')
master.to_csv(os.path.join(base_path, 'master_5m.csv'), index=False)
