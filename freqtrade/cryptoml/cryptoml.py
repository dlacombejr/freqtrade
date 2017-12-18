import os
import glob
import json
import pickle
import requests
import numpy as np


class CrytpoML(object):

    def __init__(self, whitelist):

        self.name = 'cryptoml'
        self.url = "https://cryptoml.azure-api.net/prediction/threshold/"
        self.headers = {
            'content-type': "application/json",
            'ocp-apim-subscription-key': "95e10bd93a4d4f82906e4c48a781796e",
        }
        self.payload = '{}'
        self.whitelist = whitelist
        self.buy_signal_buffer = {}
        for market in whitelist:
            self.buy_signal_buffer[market] = {
                'buy_count': 0,
                'predicted_returns': [],
                'running_return': 0
            }

    def reset_buffer(self):

        self.buy_signal_buffer = {}
        for market in self.whitelist:
            self.buy_signal_buffer[market] = {
                'buy_count': 0,
                'predicted_returns': [],
                'running_return': 0
            }

    def get_cryptoml_prediction(self, threshold=0.025):

        return requests.request(
            method="POST",
            url=self.url + str(threshold),
            headers=self.headers,
            data=self.payload
        )

    def update_buy_buffer(self, threshold=0.025):

        resp = self.get_cryptoml_prediction(threshold).json()

        # print(json.dumps(resp, indent=4))

        # for token in resp:
        #
        #     if 'BTC_' + token in self.buy_signal_buffer:
        #
        #         self.buy_signal_buffer['BTC_' + token]['buy_count'] += 1
        #         self.buy_signal_buffer['BTC_' + token]['predicted_returns'].append(resp[token])
        #         self.buy_signal_buffer['BTC_' + token]['running_return'] = np.asarray(
        #             self.buy_signal_buffer['BTC_' + token]['predicted_returns']
        #         ).mean()

        for market in self.buy_signal_buffer:

            coin = market[4:]
            if coin in resp:

                if resp[coin]:

                    self.buy_signal_buffer[market]['buy_count'] += 1
                    self.buy_signal_buffer[market]['predicted_returns'].append(resp[coin][0]['score'])
                    self.buy_signal_buffer[market]['running_return'] = np.asarray(
                        self.buy_signal_buffer[market]['predicted_returns']
                    ).mean()
                else:
                    self.buy_signal_buffer[market]['buy_count'] = 0
                    self.buy_signal_buffer[market]['predicted_returns'] = []
                    self.buy_signal_buffer[market]['running_return'] = 0

    def get_buy_signal(self, whitelist, update=True, threshold=0.025, repeats=3):

        # optionally update buy buffer
        if update:
            self.update_buy_buffer(threshold)
            print(json.dumps(self.buy_signal_buffer, indent=4))

        # get token with highest predicted return
        token = None
        expected_return = 0
        for market in self.buy_signal_buffer:
            if self.buy_signal_buffer[market]['buy_count'] >= repeats:
                if self.buy_signal_buffer[market]['running_return'] > expected_return and market in whitelist:
                    token = market
                    expected_return = self.buy_signal_buffer[market]['running_return']

        return token


class DanML(object):

    def __init__(self):

        self.name = 'danml'

        # load pipeline
        model_path = os.path.join(
            os.path.expanduser('~'),
            'freqtrade',
            'ml_dev',
            'models'
        )
        list_of_files = glob.glob(os.path.join(model_path, '*.pkl'))
        latest_file = max(list_of_files, key=os.path.getctime)
        print(latest_file)
        with open(latest_file, 'rb') as f:
            pipeline = pickle.load(f, encoding='latin1')
        f.close()

        # get model and scaler
        self.model = pipeline['model']
        self.scaler = pipeline['scaler']

    def df_preprocess(self, df):

        df = df.drop('date', axis=1)
        x = df.as_matrix()
        x = x[-1, :]
        x = x.reshape(1, -1)
        x = self.scaler.transform(x)

        return x
