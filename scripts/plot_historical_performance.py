import json
from freqtrade.persistence import Trade
from freqtrade.main import *
from typing import Dict, Optional, List



def init(config: dict, db_url: Optional[str] = None) -> None:
    """
    Initializes all modules and updates the config
    :param config: config as dict
    :param db_url: database connector string for sqlalchemy (Optional)
    :return: None
    """
    # Initialize all modules
    persistence.init(config, db_url)


with open('../config_cryptoml.json') as file:
    _CONF = json.load(file)
init(_CONF)

trades = Trade.query.filter().all()
print(trades)
