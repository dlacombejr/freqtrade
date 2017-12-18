import requests


session = requests.session()
# session.auth = ("aa81672bd8f90199763ce6ce9c4e411c", "f64508b789f2964825758a8e9029f785")
data = requests.get('https://api.hitbtc.com/api/2/public/candles/ETHBTC', params={
            'limit': 1000,
            'period': 'M1',
        }).json()

# data = requests.get('https://api.hitbtc.com/api/2/public/candles/symbol').json()
# data = requests.get('https://api.hitbtc.com/api/2/public/candles/candles/ETHBTC?period=M1').json()

print(data[-1])
print(data[-1]['timestamp'].minute)
