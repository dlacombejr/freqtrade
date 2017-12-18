#!/usr/bin/env python3
import copy
import json
import logging
import time
import argparse
import traceback
from datetime import datetime
from typing import Dict, Optional, List
from signal import signal, SIGINT, SIGABRT, SIGTERM

import requests
from cachetools import cached, TTLCache
from jsonschema import validate

from freqtrade import __version__, exchange, persistence
from freqtrade.analyze import get_buy_signal, get_ema_signal
from freqtrade.misc import CONF_SCHEMA, State, get_state, update_state
from freqtrade.persistence import Trade
from freqtrade.rpc import telegram
from freqtrade.cryptoml.cryptoml import CrytpoML, DanML

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_CONF = {}

analyzer = None

dynamic_whitelist = False


def get_quote_token(pair_):

    return pair_[4:]


def refresh_whitelist(whitelist: Optional[List[str]] = None) -> None:
    """
    Check wallet health and remove pair from whitelist if necessary
    :param whitelist: a new whitelist (optional)
    :return: None
    """
    whitelist = whitelist or _CONF['exchange']['pair_whitelist']

    sanitized_whitelist = []
    health = exchange.get_wallet_health()
    for status in health:
        if exchange.get_name() == 'HitBTC':
            pair = status['Currency']
        else:
            pair = '{}_{}'.format(_CONF['stake_currency'], status['Currency'])
        if pair not in whitelist:
            continue
        if status['IsActive']:
            sanitized_whitelist.append(pair)
        else:
            logger.info(
                'Ignoring %s from whitelist (reason: %s).',
                pair, status.get('Notice') or 'wallet is not active'
            )
    if _CONF['exchange']['pair_whitelist'] != sanitized_whitelist:
        logger.debug('Using refreshed pair whitelist: %s ...', sanitized_whitelist)
        _CONF['exchange']['pair_whitelist'] = sanitized_whitelist

        if analyzer.name == 'cryptoml':
            analyzer.whitelist = sanitized_whitelist
            analyzer.reset_buffer()


def _process(buying: bool) -> bool:
    """
    Queries the persistence layer for open trades and handles them,
    otherwise a new trade is created.
    :return: True if a trade has been created or closed, False otherwise
    """
    state_changed = False
    try:

        # Refresh whitelist based on wallet maintenance
        refresh_whitelist(
            gen_pair_whitelist(_CONF['stake_currency']) if dynamic_whitelist else None
        )
        # Query trades from persistence layer
        trades = Trade.query.filter(Trade.is_open.is_(True)).all()

        # print(len(trades))

        if buying:
            if len(trades) < _CONF['max_open_trades']:
                try:
                    # Create entity and execute trade
                    trade = create_trade(float(_CONF['stake_amount']))
                    if trade:
                        Trade.session.add(trade)
                        state_changed = True
                    else:
                        logging.info('Got no buy signal...')
                except ValueError:
                    logger.exception('Unable to create trade')

            elif analyzer.name == 'cryptoml':
                # analyzer.update_buy_buffer()
                _ = analyzer.get_buy_signal([], update=True, threshold=0.01, repeats=3)

        for trade in trades:
            # Get order details for actual price per unit
            if trade.open_order_id:
                # Update trade with order values
                logger.info('Got open order for %s', trade)
                trade.update(exchange.get_order(trade.open_order_id))

            if not close_trade_if_fulfilled(trade):
                # Check if we can sell our current pair
                state_changed = handle_trade(trade) or state_changed

            Trade.session.flush()
    except (requests.exceptions.RequestException, json.JSONDecodeError) as error:
        msg = 'Got {} in _process(), retrying in 30 seconds...'.format(error.__class__.__name__)
        logger.exception(msg)
        time.sleep(30)
    except RuntimeError:
        telegram.send_msg('*Status:* Got RuntimeError:\n```\n{traceback}```{hint}'.format(
            traceback=traceback.format_exc(),
            hint='Issue `/start` if you think it is safe to restart.'
        ))
        logger.exception('Got RuntimeError. Stopping trader ...')
        update_state(State.STOPPED)
    return state_changed


def close_trade_if_fulfilled(trade: Trade) -> bool:
    """
    Checks if the trade is closable, and if so it is being closed.
    :param trade: Trade
    :return: True if trade has been closed else False
    """
    # If we don't have an open order and the close rate is already set,
    # we can close this trade.
    if trade.close_profit is not None \
            and trade.close_date is not None \
            and trade.close_rate is not None \
            and trade.open_order_id is None:
        trade.is_open = False
        logger.info('No open orders found and trade is fulfilled. Marking %s as closed ...', trade)
        return True
    return False


def execute_sell(trade: Trade, limit: float) -> None:
    """
    Executes a limit sell for the given trade and limit
    :param trade: Trade instance
    :param limit: limit rate for the sell order
    :return: None
    """

    # check how much is available
    available = exchange.get_available_balance(get_quote_token(str(trade.pair)))

    # Execute sell and update trade record
    order_id = exchange.sell(str(trade.pair), limit, available)  # trade.amount
    trade.open_order_id = order_id

    if available < trade.amount:

        # cancel the rest
        # todo: check if wallet is healthy before cancelling
        health = exchange.get_wallet_health()
        if exchange.get_name() == 'HitBTC':
            token = trade.pair
        else:
            token = get_quote_token(trade.pair)
        token_healthy = False
        for status in health:
            if status['Currency'] == token:
                token_healthy = status['IsActive']
        if token_healthy:
            exchange.cancel_order(trade.id)

    fmt_exp_profit = round(trade.calc_profit(limit) * 100, 2)
    message = '*{}:* Selling [{}]({}) with limit `{:.8f} (profit: ~{:.2f}%)`'.format(
        trade.exchange,
        trade.pair.replace('_', '/'),
        exchange.get_pair_detail_url(trade.pair),
        limit,
        fmt_exp_profit
    )
    logger.info(message)
    telegram.send_msg(message)
    # Trade.session.flush()


def should_sell(trade: Trade, current_rate: float, current_time: datetime) -> bool:
    """
    Based an earlier trade and current price and configuration, decides whether bot should sell
    :return True if bot should sell at current rate
    """

    # todo: query model to see if more gains are possible

    # get current profit
    current_profit = trade.calc_profit(current_rate)

    # check how much is available and determine if above dust trade
    available = exchange.get_available_balance(get_quote_token(str(trade.pair)))
    satoshi = 1e-08
    dust_trade_sat = 50000 * satoshi
    trade_amount_btc = available * current_rate
    if trade_amount_btc < dust_trade_sat:
        logger.debug('Trade is DUST')
        logger.debug('Threshold not reached. (cur_profit: %1.2f%%)', current_profit * 100.0)
        logger.debug('Current rate: {}'.format(current_rate))
        return False

    # determine if stoploss is hit
    if 'stoploss' in _CONF and current_profit < float(_CONF['stoploss']):
        logger.debug('Stop loss hit.')
        return True

    # check model (before minimal rois)
    if analyzer.name == 'danml':
        # check if ema-5 is greater than last minute to continue
        # todo: also make sure previous is lower than current
        if get_ema_signal(trade.pair, 1, exchange.get_name()):
            if get_buy_signal(trade.pair, exchange.get_name(), analyzer):
                logger.debug('Threshold not reached. (cur_profit: %1.2f%%)', current_profit * 100.0)
                logger.debug('Current rate: {}'.format(current_rate))
                return False

    # check minimal rois
    for duration, threshold in sorted(_CONF['minimal_roi'].items()):
        # Check if time matches and current rate is above threshold
        time_diff = (current_time - trade.open_date).total_seconds() / 60
        if time_diff > float(duration) and current_profit > threshold:
            return True

    logger.debug('Threshold not reached. (cur_profit: %1.2f%%)', current_profit * 100.0)
    logger.debug('Current rate: {}'.format(current_rate))
    return False


def handle_trade(trade: Trade) -> bool:
    """
    Sells the current pair if the threshold is reached and updates the trade record.
    :return: True if trade has been sold, False otherwise
    """
    if not trade.is_open:
        raise ValueError('attempt to handle closed trade: {}'.format(trade))

    logger.debug('Handling %s ...', trade)

    # first determine if buy order that is still open
    if trade.open_order_id:
        order_info = exchange.get_order(trade.open_order_id)
        order_type = order_info['type']
        if order_type == 'LIMIT_BUY':
            max_open_time = 10
            current_time = datetime.utcnow()
            amount_minutes_open = (current_time - trade.open_date).total_seconds() / 60.
            if amount_minutes_open > max_open_time:
                health = exchange.get_wallet_health()
                if exchange.get_name() == 'HitBTC':
                    token = trade.pair
                else:
                    token = get_quote_token(trade.pair)
                token_healthy = False
                for status in health:
                    if status['Currency'] == token:
                        token_healthy = status['IsActive']
                if token_healthy:
                    logger.debug('Cancelling %s ...', trade)
                    exchange.cancel_order(trade.open_order_id)
                    # trade.is_open = 0
                    # trade.open_order_id = None
                    Trade.session.delete(trade)
                else:
                    logger.debug('Cancelling could not execute due to wallet heath for %s ...', trade)
                return False

    current_rate = exchange.get_ticker(trade.pair)['bid']  # ask?
    if current_rate is None:
        return False
    if should_sell(trade, current_rate, datetime.utcnow()):
        execute_sell(trade, current_rate)
        return True
    return False


def get_target_bid(ticker: Dict[str, float]) -> float:
    """ Calculates bid target between current ask price and last price """
    if ticker['ask'] < ticker['last']:
        return ticker['ask']
    balance = _CONF['bid_strategy']['ask_last_balance']
    return ticker['ask'] + balance * (ticker['last'] - ticker['ask'])


def create_trade(stake_amount: float) -> Optional[Trade]:
    """
    Checks the implemented trading indicator(s) for a randomly picked pair,
    if one pair triggers the buy_signal a new trade record gets created
    :param stake_amount: amount of btc to spend
    """
    logger.info('Creating new trade with stake_amount: %f ...', stake_amount)
    whitelist = copy.deepcopy(_CONF['exchange']['pair_whitelist'])
    # Check if stake_amount is fulfilled
    if exchange.get_balance(_CONF['stake_currency']) < stake_amount:
        raise ValueError(
            'stake amount is not fulfilled (currency={})'.format(_CONF['stake_currency'])
        )

    # Remove currently opened and latest pairs from whitelist
    for trade in Trade.query.filter(Trade.is_open.is_(True)).all():
        if trade.pair in whitelist:
            whitelist.remove(trade.pair)
            logger.debug('Ignoring %s in pair whitelist', trade.pair)
    # if not whitelist:
    #     raise ValueError('No pair in whitelist')

    # Pick pair based on StochRSI buy signals
    if analyzer.name == 'danml':
        for _pair in whitelist:
            if get_buy_signal(_pair, exchange.get_name(), analyzer):
                pair = _pair
                break
        else:
            return None
    elif analyzer.name == 'cryptoml':
        update = False
        if datetime.utcnow().minute % 5 == 0:
            update = True
        pair = analyzer.get_buy_signal(whitelist, update=update, threshold=0.01, repeats=3)
        if pair is None:
            return None

    # Calculate amount and subtract fee
    fee = exchange.get_fee()
    buy_limit = get_target_bid(exchange.get_ticker(pair))
    amount = (1 - fee) * stake_amount / buy_limit

    health = exchange.get_wallet_health()
    if exchange.get_name() == 'HitBTC':
        token = pair
    else:
        token = get_quote_token(pair)
    token_healthy = False
    for status in health:
        if status['Currency'] == token:
            token_healthy = status['IsActive']
    if token_healthy:
        order_id = exchange.buy(pair, buy_limit, amount)
        # Create trade entity and return
        message = '*{}:* Buying [{}]({}) with limit `{:.8f}`'.format(
            exchange.get_name().upper(),
            pair.replace('_', '/'),
            exchange.get_pair_detail_url(pair),
            buy_limit
        )
        logger.info(message)
        telegram.send_msg(message)
        # Fee is applied twice because we make a LIMIT_BUY and LIMIT_SELL
        return Trade(pair=pair,
                     stake_amount=stake_amount,
                     amount=amount,
                     fee=fee * 2.,
                     open_rate=buy_limit,
                     open_date=datetime.utcnow(),
                     exchange=exchange.get_name().upper(),
                     open_order_id=order_id,
                     # open_order_type='buy'
                     )
        # Trade.session.add(trade)
        # Trade.session.flush()
        # return True


def init(config: dict, db_url: Optional[str] = None) -> None:
    """
    Initializes all modules and updates the config
    :param config: config as dict
    :param db_url: database connector string for sqlalchemy (Optional)
    :return: None
    """
    # Initialize all modules
    telegram.init(config)
    persistence.init(config, db_url)
    exchange.init(config)

    # Set initial application state
    initial_state = config.get('initial_state')
    if initial_state:
        update_state(State[initial_state.upper()])
    else:
        update_state(State.STOPPED)

    # Register signal handlers
    for sig in (SIGINT, SIGTERM, SIGABRT):
        signal(sig, cleanup)


@cached(TTLCache(maxsize=1, ttl=1800))
def gen_pair_whitelist(base_currency: str, topn: int = 20, key: str = 'BaseVolume') -> List[str]:
    """
    Updates the whitelist with with a dynamically generated list
    :param base_currency: base currency as str
    :param topn: maximum number of returned results
    :param key: sort key (defaults to 'BaseVolume')
    :return: List of pairs
    """
    summaries = sorted(
        (s for s in exchange.get_market_summaries() if s['MarketName'].startswith(base_currency)),
        key=lambda s: s.get(key) or 0.0,
        reverse=True
    )
    return [s['MarketName'].replace('-', '_') for s in summaries[:topn]]


def cleanup(*args, **kwargs) -> None:
    """
    Cleanup the application state und finish all pending tasks
    :return: None
    """
    telegram.send_msg('*Status:* `Stopping trader...`')
    logger.info('Stopping trader and cleaning up modules...')
    update_state(State.STOPPED)
    persistence.cleanup()
    telegram.cleanup()
    exit(0)


def main(config_suffix=None):
    """
    Loads and validates the config and handles the main loop
    :return: None
    """
    logger.info('Starting freqtrade %s', __version__)

    global _CONF
    with open('config' + config_suffix + '.json') as file:
        _CONF = json.load(file)

    logger.info('Validating configuration ...')
    validate(_CONF, CONF_SCHEMA)

    init(_CONF)

    # get trade interval from config for main loop
    trade_interval = _CONF['trade_interval']

    # get analyze_method from config for buy signals
    analyze_method = _CONF['analyze_method']
    global analyzer
    if analyze_method == 'danml':
        analyzer = DanML()
    elif analyze_method == 'cryptoml':
        analyzer = CrytpoML(whitelist=_CONF['exchange']['pair_whitelist'])
    else:
        assert analyzer is not None

    old_state = get_state()
    logger.info('Initial State: %s', old_state)
    telegram.send_msg('*Status:* `{}`'.format(old_state.name.lower()))
    while True:

        s = datetime.utcnow().second
        m = datetime.utcnow().minute
        h = datetime.utcnow().hour

        if s == 0 and m % trade_interval == 0:

            new_state = get_state()
            # Log state transition
            if new_state != old_state:
                telegram.send_msg('*Status:* `{}`'.format(new_state.name.lower()))
                logging.info('Changing state to: %s', new_state.name)

            if new_state == State.STOPPED:
                time.sleep(1)
            elif new_state == State.RUNNING:
                _process(buying=True)
                # We need to sleep here because otherwise we would run into bittrex rate limit
                # time.sleep(exchange.get_sleep_time())
            old_state = new_state

        if s == 30:

            new_state = get_state()
            # Log state transition
            if new_state != old_state:
                telegram.send_msg('*Status:* `{}`'.format(new_state.name.lower()))
                logging.info('Changing state to: %s', new_state.name)

            if new_state == State.STOPPED:
                time.sleep(1)
            elif new_state == State.RUNNING:
                _process(buying=False)
                # We need to sleep here because otherwise we would run into bittrex rate limit
                # time.sleep(exchange.get_sleep_time())
            old_state = new_state

        if h == 0:
            if analyzer.name == 'cryptoml':
                analyzer.reset_buffer()

        else:

            time.sleep(0.1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_suffix",
                        help="suffix for config file; e.g., _cryptoml",
                        action="store", required=False, default='')
    args = parser.parse_args()

    main(config_suffix=args.config_suffix)
