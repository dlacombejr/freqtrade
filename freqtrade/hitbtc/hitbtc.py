import time
import hmac
import hashlib

try:
    from urllib import urlencode
    from urlparse import urljoin
except ImportError:
    from urllib.parse import urlencode
    from urllib.parse import urljoin

try:
    from Crypto.Cipher import AES
except ImportError:
    encrypted = False
else:
    import getpass
    import ast
    import json

    encrypted = True

import requests


API_V2 = '2'
BASE_URL_V2 = 'https://api.hitbtc.com/api/2{path}?'

PROTECTION_PUB = 'pub'  # public methods
PROTECTION_PRV = 'prv'  # authenticated methods


def encrypt(api_key, api_secret, export=True, export_fn='secrets.json'):
    salt = getpass.getpass(
        'Input encryption password (string will not show)')
    key32 = "".join([' ' if i >= len(salt) else salt[i] for i in range(32)])
    bkey32 = key32.encode('utf-8')
    cipher = AES.new(bkey32)
    api_key_n = cipher.encrypt(api_key)
    # print(type(api_key_n))
    api_secret_n = cipher.encrypt(api_secret)
    api = {'key': api_key_n.decode('latin-1'), 'secret': api_secret_n.decode('latin-1')}
    # print(api)
    if export:
        with open(export_fn, 'wb') as outfile:
            json.dump(api, outfile, indent=4)
    return api


def pretty_print_POST(req):
    """
    At this point it is completely built and ready
    to be fired; it is "prepared".

    However pay attention at the formatting used in
    this function because it is programmed to be pretty
    printed and may differ from the actual request.
    """
    print('{}\n{}\n{}\n\n{}'.format(
        '-----------START-----------',
        req.method + ' ' + req.url,
        '\n'.join('{}: {}'.format(k, v) for k, v in req.headers.items()),
        req.body,
    ))


def using_requests(request_url, apisign, request_type='GET', user=None, pswd=None):
    r = None
    if request_type == 'GET':
        r = requests.get(
            request_url,
            headers={"apisign": apisign},
            auth=(user, pswd)
        ).json()
    elif request_type == 'POST':
        r = requests.post(
            request_url,
            headers={"apisign": apisign},
            auth=(user, pswd)
        ).json()
    elif request_type == 'DELETE':
        r = requests.delete(
            request_url,
            headers={"apisign": apisign},
            auth=(user, pswd)
        ).json()
    pretty_print_POST(r.prepared())
    return r


class HitBTC(object):
    """
    Used for requesting HitBTC with API key and API secret
    """

    def __init__(self, api_key, api_secret, calls_per_second=1, dispatch=using_requests, api_version=API_V2):
        self.api_key = str(api_key) if api_key is not None else ''
        self.api_secret = str(api_secret) if api_secret is not None else ''
        self.dispatch = dispatch
        self.call_rate = 1.0 / calls_per_second
        self.last_call = None
        self.api_version = api_version

    def decrypt(self):
        if encrypted:
            cipher = AES.new(getpass.getpass(
                'Input decryption password (string will not show)'))
            try:
                if isinstance(self.api_key, str):
                    self.api_key = ast.literal_eval(self.api_key)
                if isinstance(self.api_secret, str):
                    self.api_secret = ast.literal_eval(self.api_secret)
            except Exception:
                pass
            self.api_key = cipher.decrypt(self.api_key).decode()
            self.api_secret = cipher.decrypt(self.api_secret).decode()
        else:
            raise ImportError('"pycrypto" module has to be installed')

    def wait(self):
        if self.last_call is None:
            self.last_call = time.time()
        else:
            now = time.time()
            passed = now - self.last_call
            if passed < self.call_rate:
                # print("sleep")
                time.sleep(self.call_rate - passed)

            self.last_call = time.time()

    def _api_query(self, protection=None, path_dict=None, options=None, request_type='GET'):
        """
        Queries HitBTC

        :param request_url: fully-formed URL to request
        :type options: dict
        :return: JSON response from HitBTC
        :rtype : dict
        """

        if not options:
            options = {}

        if self.api_version not in path_dict:
            raise Exception('method call not available under API version {}'.format(self.api_version))

        request_url = BASE_URL_V2
        request_url = request_url.format(path=path_dict[self.api_version])

        nonce = str(int(time.time() * 1000))

        user = None
        pswd = None
        if protection != PROTECTION_PUB:
            request_url = "{0}apikey={1}&nonce={2}&".format(request_url, self.api_key, nonce)
            user = self.api_key
            pswd = self.api_secret

        request_url += urlencode(options)

        try:
           apisign = hmac.new(self.api_secret.encode(),
                              request_url.encode(),
                              hashlib.sha512).hexdigest()

           self.wait()

           return self.dispatch(request_url, apisign, request_type, user=user, pswd=pswd)

        except:
            return {
               'success' : False,
               'message' : 'NO_API_RESPONSE',
               'result'  : None
            }

    def get_currencies(self, currency=''):
        """
        Used to get the open and available trading currencies
        at HitBTC along with other meta data.

        2.0 Endpoint: /api/2/public/currency

        Example ::
            [
               {
                  "id": "BTC",
                  "fullName": "Bitcoin",
                  "crypto": true,
                  "payinEnabled": true,
                  "payinPaymentId": false,
                  "payinConfirmations": 2,
                  "payoutEnabled": true,
                  "payoutIsPaymentId": false,
                  "transferEnabled": true
               },
               {
                  "id": "ETH",
                  "fullName": "Ethereum",
                  "crypto": true,
                  "payinEnabled": true,
                  "payinPaymentId": false,
                  "payinConfirmations": 2,
                  "payoutEnabled": true,
                  "payoutIsPaymentId": false,
                  "transferEnabled": true
               }
            ]

        :return: Available market info in JSON
        :rtype : dict
        """
        return self._api_query(path_dict={
            API_V2: '/public/currency/{currency}'.format(currency=currency)
        }, protection=PROTECTION_PUB)

    def get_symbol(self, symbol=''):
        """
        Used to get the open and available trading currencies
        at HitBTC along with other meta data.

        2.0 Endpoint: /api/2/public/currency

        Example ::
            [
              {
                "id": "ETHBTC",
                "baseCurrency": "ETH",
                "quoteCurrency": "BTC",
                "quantityIncrement": "0.001",
                "tickSize": "0.000001",
                "takeLiquidityRate": "0.001",
                "provideLiquidityRate": "-0.0001",
                "feeCurrency": "BTC"
              }
            ]

        :return: Available market info in JSON
        :rtype : list
        """
        return self._api_query(path_dict={
            API_V2: '/public/symbol/{symbol}'.format(symbol=symbol)
        }, protection=PROTECTION_PUB)

    def get_tickers(self, symbol=''):
        """
        Used to get the open and available trading currencies
        at HitBTC along with other meta data.

        2.0 Endpoint: /api/2/public/currency

        Example ::
            [
              {
                "ask": "0.050043",
                "bid": "0.050042",
                "last": "0.050042",
                "open": "0.047800",
                "low": "0.047052",
                "high": "0.051679",
                "volume": "36456.720",
                "volumeQuote": "1782.625000",
                "timestamp": "2017-05-12T14:57:19.999Z",
                "symbol": "ETHBTC"
              }
            ]
        :return: Available market info in JSON
        :rtype : list
        """
        return self._api_query(path_dict={
            API_V2: '/public/ticker/{symbol}'.format(symbol=symbol)
        }, protection=PROTECTION_PUB)

    def get_trades(self, symbol=''):
        """
        Used to get the open and available trading currencies
        at HitBTC along with other meta data.

        2.0 Endpoint: /api/2/public/currency

        Example ::
            [
              {
                "id": 9533117,
                "price": "0.046001",
                "quantity": "0.220",
                "side": "sell",
                "timestamp": "2017-04-14T12:18:40.426Z"
              },
              {
                "id": 9533116,
                "price": "0.046002",
                "quantity": "0.022",
                "side": "buy",
                "timestamp": "2017-04-14T11:56:37.027Z"
              }
            ]
        :return: Available market info in JSON
        :rtype : list
        """
        return self._api_query(path_dict={
            API_V2: '/public/trades/{symbol}'.format(symbol=symbol)
        }, protection=PROTECTION_PUB)

    def get_orderbook(self, symbol='', limit=100):
        """
        Used to get the open and available trading currencies
        at HitBTC along with other meta data.

        2.0 Endpoint: /api/2/public/currency

        Example ::
            {
              "ask": [
                {
                  "price": "0.046002",
                  "size": "0.088"
                },
                {
                  "price": "0.046800",
                  "size": "0.200"
                }
              ],
              "bid": [
                {
                  "price": "0.046001",
                  "size": "0.005"
                },
                {
                  "price": "0.046000",
                  "size": "0.200"
                }
              ]
            }
        :return: Available market info in JSON
        :rtype : dict
        """
        return self._api_query(path_dict={
            API_V2: '/public/orderbook/{symbol}'.format(symbol=symbol)
        }, options={'limit': limit}, protection=PROTECTION_PUB)

    def get_candles(self, symbol='', limit=100, period='M30'):
        """
        Used to get the open and available trading currencies
        at HitBTC along with other meta data.

        2.0 Endpoint: /api/2/public/currency

        Example ::
            [
              {
                "timestamp": "2017-10-20T20:00:00.000Z",
                "open": "0.050459",
                "close": "0.050087",
                "min": "0.050000",
                "max": "0.050511",
                "volume": "1326.628",
                "volumeQuote": "66.555987736"
              },
              {
                "timestamp": "2017-10-20T20:30:00.000Z",
                "open": "0.050108",
                "close": "0.050139",
                "min": "0.050068",
                "max": "0.050223",
                "volume": "87.515",
                "volumeQuote": "4.386062831"
              }
            ]
        :return: Available market info in JSON
        :rtype : list
        """
        return self._api_query(path_dict={
            API_V2: '/public/candles/{symbol}'.format(symbol=symbol)
        }, options={'limit': limit, 'period': period}, protection=PROTECTION_PUB)

    def get_balances(self):
        """
        Used to retrieve all balances from your account.

        Endpoint:
        1.1 /account/getbalances
        2.0 /key/balance/getbalances

        Example ::
            [
              {
                "currency": "ETH",
                "available": "10.000000000",
                "reserved": "0.560000000"
              },
              {
                "currency": "BTC",
                "available": "0.010205869",
                "reserved": "0"
              }
            ]

        :return: Balances info in JSON
        :rtype : list
        """
        return self._api_query(path_dict={
            API_V2: '/trading/balance',
        }, protection=PROTECTION_PRV)

    def get_open_orders(self):
        """
        Get all orders that you currently have opened.
        A specific market can be requested.

        Endpoint:
        1.1 /market/getopenorders
        2.0 /key/market/getopenorders

        :param market: String literal for the market (ie. BTC-LTC)
        :type market: str
        :return: Open orders info in JSON
        :rtype : dict
        """
        return self._api_query(path_dict={
            API_V2: '/order'
        }, protection=PROTECTION_PRV)

    def get_order(self, uuid=''):
        """
        Used to get details of buy or sell order

        Endpoint:
        1.1 /account/getorder
        2.0 /key/orders/getorder

        :param uuid: uuid of buy or sell order
        :type uuid: str
        :return:
        :rtype : dict

        Example:

            {
              "id": 0,
              "clientOrderId": "d8574207d9e3b16a4a5511753eeef175",
              "symbol": "ETHBTC",
              "side": "sell",
              "status": "canceled",
              "type": "limit",
              "timeInForce": "GTC",
              "quantity": "0.000",
              "price": "0.046016",
              "cumQuantity": "0.000",
              "createdAt": "2017-05-15T17:01:05.092Z",
              "updatedAt": "2017-05-15T18:08:57.226Z"
            }

        """
        return self._api_query(path_dict={
            API_V2: '/order/{clientOrderId}'.format(clientOrderId=uuid)
        }, protection=PROTECTION_PRV)

    def create_new_order(self, symbol, side, type, timeInForce, quantity, price, strictValidate=True):
        """
        Used to get details of buy or sell order

        Endpoint:
        1.1 /account/getorder
        2.0 /key/orders/getorder

        :param uuid: uuid of buy or sell order
        :type uuid: str
        :return:
        :rtype : dict
        """
        return self._api_query(path_dict={
            API_V2: '/order'
        }, options={
            'symbol': symbol,
            'side': side,
            'type': type,
            'timeInForce': timeInForce,
            'quantity': quantity,
            'price': price,
            'strictValidate': strictValidate
        },
            protection=PROTECTION_PRV,
            request_type='POST'
        )

    def cancel_order(self, uuid=''):
        """
        Used to get details of buy or sell order

        Endpoint:
        1.1 /account/getorder
        2.0 /key/orders/getorder

        :param uuid: uuid of buy or sell order
        :type uuid: str
        :return:
        :rtype : dict
        """
        return self._api_query(path_dict={
            API_V2: '/order/{clientOrderId}'.format(clientOrderId=uuid)
        }, protection=PROTECTION_PRV, request_type='DELETE')

    def get_trading_commission(self, symbol='ETHBTC'):
        """
        Used to get details of buy or sell order

        Endpoint:
        1.1 /account/getorder
        2.0 /key/orders/getorder

        :param uuid: uuid of buy or sell order
        :type uuid: str
        :return:
        :rtype : dict
        """
        return self._api_query(path_dict={
            API_V2: '/trading/fee/{symbol}'.format(symbol=symbol)
        }, protection=PROTECTION_PRV, request_type='GET')

