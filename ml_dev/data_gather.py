import os
import time
from datetime import datetime
from freqtrade.hitbtc.hitbtc import HitBTC
from freqtrade.analyze import parse_ticker_dataframe, populate_indicators

# create exchange api session
hitbtc = HitBTC(api_key=None, api_secret=None)

# VARIALBES
trade_interval = 5
market = 'BCHBTC'

# ensure directory exists
base_path = os.path.join(os.path.expanduser('~'), 'freqtrade')
data_path = os.path.join(base_path, 'ml_dev', 'data', 'hitbtc')
market_path = os.path.join(data_path, market, str(trade_interval) + 'min')
if not os.path.isdir(market_path):
    os.mkdir(market_path)
assert trade_interval == 1 or trade_interval == 5

# gather data
while True:

    # check time
    t = datetime.utcnow()
    s = t.second
    m = t.minute

    if s == 0 and m % trade_interval == 0:

        # get OHLC candles
        period = 'M' + str(trade_interval)
        resp = hitbtc.get_candles(market, limit=1000, period=period)

        # parse response into data frame and add technical indicators
        df = parse_ticker_dataframe(resp, exchange_name='HitBTC')
        df = populate_indicators(df)

        # add ask, bid, and last
        ticker = hitbtc.get_tickers(market)
        df['ask'] = ticker['ask']
        df['bid'] = ticker['bid']
        df['last'] = ticker['last']

        # save the last row of data frame
        file_name = list(str(df['date'][len(df) - 1]))
        file_name = "".join(file_name).replace(' ', '_').replace(':', '-') + '.csv'
        save_path = os.path.join(market_path, file_name)
        df.tail(1).to_csv(save_path, index=False)

    else:

        time.sleep(0.25)
