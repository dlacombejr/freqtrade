
# coding: utf-8

# In[1]:

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')


# In[3]:

import os
import sys
import time

depth = 2
module_path = os.path.abspath('.')
sys.path.append(module_path)
for d in xrange(depth):
    module_path = os.path.abspath(os.path.join(module_path, '..'))
    if module_path not in sys.path:
        sys.path.append(module_path)

# from python_bittrex.bittrex.bittrex import Bittrex


# In[4]:

get_ipython().magic(u'matplotlib inline')
from matplotlib import pylab as pl

import seaborn as sns
import pandas as pd


# In[5]:

# define paths
base_path = os.path.join(os.path.expanduser('~'), 'freqtrade')
data_path = os.path.join(base_path, 'ml_dev', 'data')


# In[377]:

import pandas as pd

df = pd.read_csv(os.path.join(data_path, 'master.csv'))


# In[378]:

print df.shape


# In[379]:

# delete the buy and buy price columns
df = df.drop('buy', axis=1)
df = df.drop('buy_price', axis=1)


# In[380]:

# delete the date column
df = df.drop('date', axis=1)


# In[381]:

print df.shape
print list(df.columns.values)


# In[382]:

data = df.as_matrix()


# In[392]:

x = data[:, :-1]
y = data[:, -1]


# In[393]:

print x.shape
print y.shape


# In[385]:

import numpy as np

# create categorical version of y
y_cat = np.copy(y)
y_cat[y_cat == 0] = 0
y_cat[y_cat > 0] = 1
y_cat[y_cat < 0] = -1


# In[386]:

## apply log scaling to one-sided heavy tailed distributions (makes most sense for ratios)
# x[:, 7:13] = np.log(np.clip(x[:, 7:13], 1e-4, np.inf))


# In[387]:

# x[:, -1] = np.log(np.clip(x[:, -1], 1e-4, np.inf))


# In[395]:

from sklearn.preprocessing import robust_scale, quantile_transform, StandardScaler
from sklearn import preprocessing

# x_robust = quantile_transform(x[:, 7:], random_state=0, n_quantiles=4, axis=0, output_distribution='normal')
# x = robust_scale(x)
x = preprocessing.scale(x)


# In[397]:

print np.mean(x, axis=0)


# In[16]:

# put data in to df for visualization
features = [
#     'price', 'ask', 'bid', 'high', 'low', 'last', 'prev_day', 
    'base_vol', 'open_buy_orders', 'open_sell_orders', 
    'ratio5', 'ratio10', 'ratio25', 
    'change5', 'change10', 'change25', 
    'stv5', 'stv10', 'stv25', 'b_a_spread', 
    'smart_price', 'spi'
]
df = pd.DataFrame()
for f in range(len(features)):
    df[features[f]] = x[:, f]
df['target_categorical'] = y_cat


# In[17]:

# visualize pairplot
sns.set(style="ticks")
sns.pairplot(df, hue="target_categorical") 


# In[34]:

# first create model that predicts if there will be change


# In[32]:

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# shuffle data and perfrom train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y_cat, test_size=1 / 3., shuffle=True)


# In[33]:

down = np.sum(y_train == -1)
no_change = np.sum(y_train == 0)
up = np.sum(y_train == 1)
total = y_train.shape[0]

print down
print no_change
print up
print total


# In[34]:

down_weight = 1 - down / float(total)
no_weight = 1 - no_change / float(total)
up_weight = 1 - up / float(total)


# In[35]:

sample_weights = np.zeros((y_train.shape[0]))
sample_weights[y_train == -1] = down_weight
sample_weights[y_train == 0] = no_weight
sample_weights[y_train == 1] = up_weight


# In[62]:

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

model = GradientBoostingClassifier(learning_rate=0.02, subsample=1, n_estimators=100, max_depth=2)
# model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train, sample_weight=sample_weights)


# In[63]:

t0 = time.time()
predictions = model.predict(X_test)
score = model.score(X_test, y_test)
print('inference time: {}'.format(time.time() - t0))
print('score: {}'.format(score))


# In[64]:

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predictions)


# In[43]:

# print 169 / float(121 + 169)
print 148 / float(148 + 94)


# In[177]:

y_sign = np.sign(y)
y_log = np.log(np.clip(np.abs(y) * 1000, 1e-6, np.inf))
y_log *= y_sign


# In[190]:

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# shuffle data and perfrom train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=1 / 3., shuffle=True)


# In[191]:

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR


# In[192]:

model = GradientBoostingRegressor(loss='ls', learning_rate=0.01, subsample=0.5, n_estimators=500)
# model = RandomForestRegressor(criterion='mae')  # 'mse'
# model = SVR(kernel='rbf', C=1e3, gamma=0.1)


# In[193]:

model.fit(X_train, y_train)


# In[194]:

t0 = time.time()
predictions = model.predict(X_test)
score = model.score(X_test, y_test)
print('inference time: {}'.format(time.time() - t0))
print('score: {}'.format(score))


# In[195]:

# pl.figure()
# pl.bar(np.arange(y_test.shape[0]), y_test, label='target')
# pl.bar(np.arange(y_test.shape[0]), predictions, label='predictions', c='red')
# pl.legend()
# # pl.axis([])
# pl.show()

pl.figure()
n_examples = 100
pl.bar(np.arange(n_examples), y_test[:n_examples], label='target')
pl.bar(np.arange(n_examples), predictions[:n_examples], label='predictions', color='red', alpha=0.5)
pl.legend()
# pl.axis([])
pl.show()


# In[196]:

pl.hist(y_test, 100, alpha=0.5);
pl.hist(predictions, 50, alpha=0.5);


# In[152]:

y_sign = np.sign(y_test)
y_log = np.log(np.clip(np.abs(y_test), 1e-6, np.inf))
y_log *= y_sign


# In[153]:

pl.hist(y_log, 100, alpha=0.5);


# In[236]:

def buy(pred, fee=0.0025):
    
    buy = False
#     print pred
    if pred == 1:
        buy = True
        
    return buy


# In[241]:

def calc_profit(y_pred, y_true, fee=0.0025):
    
    profit = 0
    for trade in range(len(y_pred)):
        
        b = buy(y_pred[trade])
        if b:
            profit += y_true[trade] - fee
#             print profit
            
    return profit


# In[140]:

0.25 /100.


# In[243]:

prop_profitable = np.sum(y > 0.0025) / float(len(y))
print prop_profitable
print 1 / prop_profitable


# In[201]:

import numpy as np

# create categorical version of y
y_cat = np.copy(y)
y_cat[y_cat > 0.0025] = 1
y_cat[y_cat <= 0.0025] = 0


# In[369]:

from sklearn.decomposition import PCA

pca = PCA(n_components=7)
pca.fit(x)
x = pca.transform(x)


# In[398]:

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# shuffle data and perfrom train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=1 / 3., shuffle=True, random_state=0)


# In[399]:

# create categorical version of y
y_train_cat = np.copy(y_train)
y_train_cat[y_train > 0.0025] = 1
y_train_cat[y_train <= 0.0025] = 0
y_test_cat = np.copy(y_test)
y_test_cat[y_test > 0.0025] = 1
y_test_cat[y_test <= 0.0025] = 0


# In[400]:

from imblearn.over_sampling import RandomOverSampler
from collections import Counter

ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_sample(X_train, y_train_cat)
print(sorted(Counter(y_resampled).items()))


# In[413]:

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn import svm

# model = GradientBoostingClassifier(learning_rate=0.02, subsample=1, n_estimators=500, max_depth=2)
# model = svm.SVC(kernel='rbf', class_weight={0: 1, 1: 1})
model = RandomForestClassifier(n_estimators=100, class_weight={0: 1, 1: 1})
model.fit(X_resampled, y_resampled)


# In[414]:

t0 = time.time()
predictions = model.predict(X_test)
score = model.score(X_test, y_test_cat)
print('inference time: {}'.format(time.time() - t0))
print('score: {}'.format(score))


# In[415]:

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test_cat, predictions)


# In[416]:

calc_profit(predictions, y_test)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[20]:

# next create model that regresses change


# In[21]:

y_change_only = y[y != 0]
x_change_only = x[y != 0]


# In[22]:

pl.hist(y_change_only, 200);


# In[47]:

y_sign = np.sign(y_change_only)
y_change_only /= np.max(np.abs(y_change_only))
y_change_only *= 100
y_change_only = np.log(np.clip(np.abs(y_change_only), 1e-6, np.inf))
y_change_only *= y_sign
y_change_only /= np.max(y_change_only)


# In[48]:

pl.hist(y_change_only, 200);


# In[23]:

below = np.sum(y_change_only > 0)
above = np.sum(y_change_only < 0)


# In[24]:

print np.minimum(below, above)


# In[25]:

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# shuffle data and perfrom train-test split
X_train, X_test, y_train, y_test = train_test_split(x_change_only, y_change_only, test_size=0.33, shuffle=True)


# In[26]:

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR


# In[27]:

model = GradientBoostingRegressor(loss='huber', learning_rate=0.1, subsample=0.5, n_estimators=500)
# model = RandomForestRegressor(criterion='mse')
# model = SVR(kernel='rbf', C=1e3, gamma=0.1)


# In[28]:

model.fit(X_train, y_train)


# In[29]:

import time


# In[30]:

t0 = time.time()
predictions = model.predict(X_test)
score = model.score(X_test, y_test)
print('inference time: {}'.format(time.time() - t0))
print('score: {}'.format(score))


# In[31]:

# pl.figure()
# pl.bar(np.arange(y_test.shape[0]), y_test, label='target')
# pl.bar(np.arange(y_test.shape[0]), predictions, label='predictions', c='red')
# pl.legend()
# # pl.axis([])
# pl.show()

pl.figure()
n_examples = 100
pl.bar(np.arange(n_examples), y_test[:n_examples], label='target')
pl.bar(np.arange(n_examples), predictions[:n_examples], label='predictions', color='red', alpha=0.5)
pl.legend()
# pl.axis([])
pl.show()


# In[542]:

pl.hist(y_test, 100, alpha=0.5);
pl.hist(predictions, 50, alpha=0.5);


# In[543]:

y_gain = y[y > 0]
x_gain = x[y > 0]


# In[544]:

# shuffle data and perfrom train-test split
X_train, X_test, y_train, y_test = train_test_split(x_gain, y_gain, test_size=0.33, shuffle=True)


# In[545]:

model = GradientBoostingRegressor(loss='huber', learning_rate=0.1, subsample=0.5, n_estimators=500)
# model = RandomForestRegressor(criterion='mse')
# model = SVR(kernel='rbf', C=1e3, gamma=0.1)


# In[546]:

model.fit(X_train, y_train)


# In[547]:

t0 = time.time()
predictions = model.predict(X_test)
score = model.score(X_test, y_test)
print('inference time: {}'.format(time.time() - t0))
print('score: {}'.format(score))


# In[548]:

# pl.figure()
# pl.bar(np.arange(y_test.shape[0]), y_test, label='target')
# pl.bar(np.arange(y_test.shape[0]), predictions, label='predictions', c='red')
# pl.legend()
# # pl.axis([])
# pl.show()

pl.figure()
n_examples = 100
pl.bar(np.arange(n_examples), y_test[:n_examples], label='target')
pl.bar(np.arange(n_examples), predictions[:n_examples], label='predictions', color='red', alpha=0.5)
pl.legend()
# pl.axis([])
pl.show()


# In[ ]:



