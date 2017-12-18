import os
import json
import time
from freqtrade import exchange
from freqtrade.analyze import analyze_ticker


# define market and time interval
market = 'BTC_ETH'
time_interval = 1

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

# gather data
while 1 == 1:

	# get ticker info
	df = analyze_ticker(pair=market, tick_interval=time_interval)

	# get the most recent row
	# data = df.iloc[-1]
	data = df.tail(1)

	# save the data frame
	file_name = list(str(data['date'][len(df) - 1]))
	file_name[10] = '_'
	file_name[13] = '-'
	file_name[16] = '-'
	file_name = "".join(file_name)
	save_path = os.path.join(market_path, file_name + '.csv')
	data.to_csv(save_path, index=False)

	time.sleep(60)
