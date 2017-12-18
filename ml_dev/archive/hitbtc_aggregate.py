import os
import pandas as pd
from freqtrade.analyze import populate_indicators

import os
import glob
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta

import arrow
import talib.abstract as ta
from pandas import DataFrame, to_datetime

from freqtrade.exchange import get_ticker_history
from freqtrade.vendor.qtpylib.indicators import awesome_oscillator


def populate_indicators(dataframe: DataFrame) -> DataFrame:
    """
    Adds several different TA indicators to the given DataFrame
    """
    dataframe['sar'] = ta.SAR(dataframe)
    dataframe['adx'] = ta.ADX(dataframe)
    stoch = ta.STOCHF(dataframe)
    dataframe['fastd'] = stoch['fastd']
    dataframe['fastk'] = stoch['fastk']
    dataframe['blower'] = ta.BBANDS(dataframe, nbdevup=2, nbdevdn=2)['lowerband']
    dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)
    dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)
    dataframe['mfi'] = ta.MFI(dataframe)
    dataframe['cci'] = ta.CCI(dataframe)
    dataframe['rsi'] = ta.RSI(dataframe)
    dataframe['mom'] = ta.MOM(dataframe)
    dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
    dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
    dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
    dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
    dataframe['ao'] = awesome_oscillator(dataframe)
    macd = ta.MACD(dataframe)
    dataframe['macd'] = macd['macd']
    dataframe['macdsignal'] = macd['macdsignal']
    dataframe['macdhist'] = macd['macdhist']

    # add volatility indicators
    dataframe['natr'] = ta.NATR(dataframe)

    # add volume indicators
    dataframe['obv'] = ta.OBV(dataframe)

    # add more momentum indicators
    dataframe['rocp'] = ta.ROCP(dataframe)

    # add some pattern recognition
    dataframe['CDL2CROWS'] = ta.CDL2CROWS(dataframe)
    dataframe['CDL3BLACKCROWS'] = ta.CDL3BLACKCROWS(dataframe)
    dataframe['CDL3INSIDE'] = ta.CDL3INSIDE(dataframe)
    dataframe['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(dataframe)
    dataframe['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(dataframe)
    dataframe['CDL3STARSINSOUTH'] = ta.CDL3STARSINSOUTH(dataframe)
    dataframe['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(dataframe)
    dataframe['CDLADVANCEBLOCK'] = ta.CDLADVANCEBLOCK(dataframe)
    dataframe['CDLBELTHOLD'] = ta.CDLBELTHOLD(dataframe)
    dataframe['CDLBREAKAWAY'] = ta.CDLBREAKAWAY(dataframe)
    dataframe['CDLDOJI'] = ta.CDLDOJI(dataframe)
    dataframe['CDLDOJISTAR'] = ta.CDLDOJISTAR(dataframe)
    dataframe['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(dataframe)
    dataframe['CDLENGULFING'] = ta.CDLENGULFING(dataframe)
    dataframe['CDLHAMMER'] = ta.CDLHAMMER(dataframe)
    dataframe['CDLBREAKAWAY'] = ta.CDLBREAKAWAY(dataframe)
    dataframe['CDLBREAKAWAY'] = ta.CDLBREAKAWAY(dataframe)

    # enter categorical time
    for h in range(24):
        dataframe['hour_{0:02}'.format(h)] = 0
    for entry in range(len(dataframe)):
        print(entry)
        hour = datetime.strptime(str(dataframe['date'][entry]), "%Y-%m-%d %H:%M:%S").hour
        for h in range(24):
            if int(h == hour) == 1:
                dataframe['hour_{0:02}'.format(h)][entry] = 1

    return dataframe



# define paths
base_path = os.path.join(os.path.expanduser('~'), 'freqtrade')
ml_path = os.path.join(base_path, 'ml_dev')
data_path = os.path.join(ml_path, 'data')

# define exchange
exchange = 'hitbtc'

# load back in
master = pd.read_csv(os.path.join('/Users/dan/python-hitbtc/data/master', 'master_ethbtc_1min_v2.csv'))
# print(pd.to_datetime(master.date, format="%m/%d/%y %H:%M"))
# master['date'] = pd.to_datetime(master.date, format="%m/%d/%y %H:%M")


# populate indicators
master = populate_indicators(master)

# # get price percentage change at t+1
# percentage_change = (master['close'][entry + 1] - master['open'][entry + 1]) / master['open'][entry + 1]
# print(percentage_change)
# master['percent_change'][entry] = percentage_change
#
# save master
# master.to_csv(os.path.join(data_path, exchange, 'master.csv'), index=False)
master.to_csv(os.path.join(data_path, exchange, 'master_ethbtc_1min_v2.csv'), index=False)
