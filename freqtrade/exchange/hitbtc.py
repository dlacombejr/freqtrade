import logging
from typing import List, Dict

import requests
from freqtrade.hitbtc.hitbtc import HitBTC as _HitBTC

from freqtrade.exchange.interface import Exchange

logger = logging.getLogger(__name__)

_API: _HitBTC = None
_EXCHANGE_CONF: dict = {}


class HitBTC(Exchange):
    """
    HitBTC API wrapper.
    """

    @property
    def sleep_time(self) -> float:
        """ Sleep time to avoid rate limits, used in the main loop """
        return 25

    def __init__(self, config: dict) -> None:
        global _API, _EXCHANGE_CONF

        _EXCHANGE_CONF.update(config)
        _API = _HitBTC(
            api_key=_EXCHANGE_CONF['key'],
            api_secret=_EXCHANGE_CONF['secret'],
            calls_per_second=5,
        )
        # _API.decrypt()

    @property
    def fee(self) -> float:
        # See https://hitbtc.com/fees-and-limits
        data = _API.get_trading_commission()
        if 'error' in data:
            raise RuntimeError('{message}'.format(
                message=data['error']['message']))
        return float(data['takeLiquidityRate']) # should be 0.001

    def buy(self, pair: str, rate: float, amount: float) -> str:
        data = _API.create_new_order(
            symbol=pair.replace('_', ''),
            side='buy',
            type='limit',
            timeInForce='GTC',
            quantity=amount,
            price=rate,
            strictValidate=True
        )
        if 'error' in data:
            raise RuntimeError('{message} params=({pair}, {rate}, {amount})'.format(
                message=data['error']['message'],
                pair=pair.replace('_', ''),
                rate=rate,
                amount=amount))
        return data['clientOrderId']

    def sell(self, pair: str, rate: float, amount: float) -> str:
        data = _API.create_new_order(
            symbol=pair.replace('_', ''),
            side='sell',
            type='limit',
            timeInForce='GTC',
            quantity=amount,
            price=rate,
            strictValidate=True
        )
        if 'error' in data:
            raise RuntimeError('{message} params=({pair}, {rate}, {amount})'.format(
                message=data['error']['message'],
                pair=pair.replace('_', ''),
                rate=rate,
                amount=amount))
        return data['clientOrderId']

    def get_balance(self, currency: str) -> float:
        data = _API.get_balances()
        if 'error' in data:
            raise RuntimeError('{message} params=({currency})'.format(
                message=data['error']['message'],
                currency=currency))
        else:
            for c in data:
                if c['currency'] == currency:
                    break
        return float(c['available'] + c['reserved'] or 0.0)

    def get_available_balance(self, currency: str) -> float:
        data = _API.get_balances()
        if 'error' in data:
            raise RuntimeError('{message} params=({currency})'.format(
                message=data['error']['message'],
                currency=currency))
        else:
            for c in data:
                if c['currency'] == currency:
                    break
        return float(c['available'] or 0.0)

    def get_balances(self):
        data = _API.get_balances()
        if 'error' in data:
            raise RuntimeError('{message}'.format(message=data['error']['message']))
        return data

    def get_ticker(self, pair: str) -> dict:
        data = _API.get_tickers(pair.replace('_', ''))
        if 'error' in data:
            raise RuntimeError('{message} params=({pair})'.format(
                message=data['error']['message'],
                pair=pair))
        return {
            'bid': float(data['bid']),
            'ask': float(data['ask']),
            'last': float(data['last']),
        }

    def get_ticker_history(self, pair: str, tick_interval: int):
        if tick_interval == 1:
            interval = 'M1'
        elif tick_interval == 5:
            interval = 'M5'
        else:
            raise ValueError('Cannot parse tick_interval: {}'.format(tick_interval))

        data = _API.get_candles(
            symbol=pair.replace('_', ''),
            limit=1000,
            period=interval
        )

        if 'error' in data:
            raise RuntimeError('{message} params=({pair})'.format(
                message=data['error']['message'],
                pair=pair))

        return data

    def get_order(self, order_id: str) -> Dict:
        data = _API.get_order(order_id)
        if 'error' in data:
            raise RuntimeError('{message} params=({order_id})'.format(
                message=data['error']['message'],
                order_id=order_id))
        return {
            'id': data['clientOrderId'],
            'type': data['type'],
            'pair': data['symbol'].replace('-', '_'),
            'opened': data['createdAt'],
            'rate': data['price'],
            'amount': data['quantity'],
            'remaining': data['cumQuantity'],
            'closed': data['status'] == 'closed',
        }

    def cancel_order(self, order_id: str) -> None:
        data = _API.cancel_order(order_id)
        if 'error' in data:
            raise RuntimeError('{message} params=({order_id})'.format(
                message=data['error']['message'],
                order_id=order_id))

    def get_pair_detail_url(self, pair: str) -> str:
        return 'https://api.hitbtc.com/api/2'

    def get_markets(self) -> List[str]:
        data = _API.get_symbol()
        if 'error' in data:
            raise RuntimeError('{message}'.format(message=data['error']['message']))
        return [m['id'].replace('-', '_') for m in data]

    def get_wallet_health(self) -> List[Dict]:
        data = _API.get_currencies()
        if 'error' in data:
            raise RuntimeError('{message}'.format(message=data['error']['message']))
        return [{
            'Currency': entry['id'] + '_BTC',
            'IsActive': entry['transferEnabled'],
            'LastChecked': False,
            'Notice': False,
        } for entry in data]
