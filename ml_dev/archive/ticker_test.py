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

# get ticker history at utc 5-minute intervals and check how long until history comes in through api
lags = {}
while True:

    # get utc time minute
    t = datetime.datetime.utcnow().minute

    # get ticker history
    df = analyze_ticker(pair=market, tick_interval=time_interval)
    ts = df['date'][len(df) - 1]

    # enter lags into dict
    if ts not in lags:

        lags[ts] = t - ts.minute

        print(lags)

    # sleep for a second
    time.sleep(25)
