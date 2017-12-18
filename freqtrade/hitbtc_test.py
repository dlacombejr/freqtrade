import os
import json
import time
import datetime
import pandas as pd
from freqtrade import exchange
from freqtrade.analyze import analyze_ticker


# ensure directory exists
base_path = os.path.join(os.path.expanduser('~'), 'freqtrade')

# get configuration
with open(os.path.join(base_path, 'config.json')) as file:
	_CONF = json.load(file)

# initialize the exchange
exchange.init(_CONF)

# get ticker
data = exchange.get_ticker(pair='ETH_BTC')
print(data)


# get ticker history
df = exchange.get_ticker_history(pair='ETH_BTC', tick_interval=1)
print(pd.DataFrame(df))

# get markets
data = exchange.get_markets()
print(data)

# get name
print(exchange.get_name())

print(exchange.get_sleep_time())
print(exchange.get_fee())
