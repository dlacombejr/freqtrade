import json
import time
import datetime
from hitbtc import HitBTC


hitbtc = HitBTC(api_key=None, api_secret=None)

print(datetime.datetime.utcnow())
resp = hitbtc.get_candles('ETHBTC', limit=10, period='M1')

print(json.dumps(resp, indent=4))
print(len(resp))
print(datetime.datetime.utcnow())

###

import pandas as pd
from pandas import to_datetime

columns = {'close': 'close', 'max': 'high', 'min': 'low', 'open': 'open', 'timestamp': 'date', 'volume': 'volume'}
df = pd.DataFrame(resp).drop('volumeQuote', 1).rename(columns=columns)
df['date'] = to_datetime(df['date'], utc=True, infer_datetime_format=True)
df.sort_values('date', inplace=True)

print(df)
