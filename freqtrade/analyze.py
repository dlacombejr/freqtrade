import logging
import pandas as pd
from datetime import datetime
from datetime import timedelta

import arrow
import talib.abstract as ta
from pandas import DataFrame, to_datetime

from freqtrade.exchange import get_ticker_history
from freqtrade.vendor.qtpylib.indicators import awesome_oscillator

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# create global variables to test predictions
n_possible_trades = 0
n_trades_made = 0


def parse_ticker_dataframe(ticker: list, exchange_name: str) -> DataFrame:
    """
    Analyses the trend for the given ticker history
    :param ticker: See exchange.get_ticker_history
    :return: DataFrame
    """
    print(exchange_name)
    if exchange_name == 'Bittrex':

        columns = {'C': 'close', 'V': 'volume', 'O': 'open', 'H': 'high', 'L': 'low', 'T': 'date'}
        frame = DataFrame(ticker) \
            .drop('BV', 1) \
            .rename(columns=columns)
        frame['date'] = to_datetime(frame['date'], utc=True, infer_datetime_format=True)
        frame.sort_values('date', inplace=True)
        return frame

    elif exchange_name == 'HitBTC':

        columns = {'close': 'close', 'max': 'high', 'min': 'low', 'open': 'open', 'timestamp': 'date', 'volume': 'volume'}
        frame = DataFrame(ticker) \
            .drop('volumeQuote', 1) \
            .rename(columns=columns)
        frame['date'] = to_datetime(frame['date'], utc=True, infer_datetime_format=True)
        frame['close'] = pd.to_numeric(frame['close'])
        frame['high'] = pd.to_numeric(frame['high'])
        frame['low'] = pd.to_numeric(frame['low'])
        frame['open'] = pd.to_numeric(frame['open'])
        frame['volume'] = pd.to_numeric(frame['volume'])
        frame.sort_values('date', inplace=True)
        return frame


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
    hour = datetime.strptime(str(dataframe['date'][len(dataframe) - 1]), "%Y-%m-%d %H:%M:%S").hour
    for h in range(24):
        dataframe['hour_{0:02}'.format(h)] = int(h == hour)

    return dataframe


def populate_buy_trend(dataframe: DataFrame, analyzer: object) -> DataFrame:
    """
    Based on TA indicators, populates the buy trend for the given dataframe
    :param dataframe: DataFrame
    :return: DataFrame with buy column
    """

    # print(dataframe['close'][len(dataframe)-1])
    # print(dataframe['sma'][len(dataframe)-1])
    # print(dataframe['tema'][len(dataframe)-1])
    # print(dataframe['blower'][len(dataframe)-1])
    # print(dataframe['mfi'][len(dataframe)-1])
    # print(dataframe['fastd'][len(dataframe)-1])
    # print(dataframe['adx'][len(dataframe)-1])
    #
    # dataframe.ix[
    #     (dataframe['close'] < dataframe['sma']) &
    #     (dataframe['tema'] <= dataframe['blower']) &
    #     (dataframe['mfi'] < 25) &
    #     (dataframe['fastd'] < 25) &
    #     (dataframe['adx'] > 30),
    #     'buy'] = 1

    dataframe_for_model = analyzer.df_preprocess(dataframe)

    p = analyzer.model.predict(dataframe_for_model)

    print(analyzer.model.predict_proba(dataframe_for_model))

    global n_possible_trades
    global n_trades_made

    # ## OVERRIDE
    # p = 1

    n_possible_trades += 1
    if p == 1:
        dataframe['buy'] = 1
        n_trades_made += 1
    else:
        dataframe['buy'] = 0
    dataframe.ix[dataframe['buy'] == 1, 'buy_price'] = dataframe['close']

    # print("number of possible trades: {} -- number of trades made {}".format(n_possible_trades, n_trades_made))

    # todo: get UTC time and check if close to it

    return dataframe


def analyze_ticker(pair: str, tick_interval: int, exchange_name: str, analyzer: object) -> DataFrame:
    """
    Get ticker data for given currency pair, push it to a DataFrame and
    add several TA indicators and buy signal to it
    :return DataFrame with ticker data and indicator data
    """
    data = get_ticker_history(pair, tick_interval)
    dataframe = parse_ticker_dataframe(data, exchange_name)

    if dataframe.empty:
        logger.warning('Empty dataframe for pair %s', pair)
        return dataframe

    dataframe = populate_indicators(dataframe)
    dataframe = populate_buy_trend(dataframe, analyzer)
    return dataframe


def get_buy_signal(pair: str, exchange_name: str, analyzer: object) -> bool:
    """
    Calculates a buy signal based several technical analysis indicators
    :param pair: pair in format BTC_ANT or BTC-ANT
    :return: True if pair is good for buying, False otherwise
    """

    dataframe = analyze_ticker(pair, 1, exchange_name, analyzer)  # 5

    if dataframe.empty:
        return False

    latest = dataframe.iloc[-1]

    # Check if dataframe is out of date
    signal_date = arrow.get(latest['date'])
    if signal_date < arrow.now() - timedelta(minutes=10):
        return False

    signal = latest['buy'] == 1
    logger.debug('buy_trigger: %s (pair=%s, signal=%s)', latest['date'], pair, signal)
    return signal

def get_ema_signal(pair: str, tick_interval: int, exchange_name: str) -> bool:
    """
    Get ticker data for given currency pair, push it to a DataFrame and
    add several TA indicators and buy signal to it
    :return DataFrame with ticker data and indicator data
    """
    data = get_ticker_history(pair, tick_interval)
    dataframe = parse_ticker_dataframe(data, exchange_name)

    if dataframe.empty:
        logger.warning('Empty dataframe for pair %s', pair)
        return False

    dataframe = populate_indicators(dataframe)
    ema5_diff = dataframe['ema5'][len(dataframe) - 1] - dataframe['ema5'][len(dataframe) - 2]
    print(ema5_diff > 0.)
    return ema5_diff > 0.
