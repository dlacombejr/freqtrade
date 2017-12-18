import os
import pandas as pd


# define market
market = 'BTC_ETH'

# define paths
base_path = os.path.join(os.path.expanduser('~'), 'freqtrade', 'ml_dev', 'data')
market_path = os.path.join(base_path, 'markets', market)

df = pd.read_csv(os.path.join(base_path, 'master.csv'))

print(df)
