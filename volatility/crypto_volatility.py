import pandas as pd
import requests
import json as json
import matplotlib.pyplot as plt
import numpy as np


# Wrapper for CryptoCompare API
def cryptocompare_history(
        base_url='https://min-api.cryptocompare.com/data/histoday',
        from_symbol = 'BTC',
        to_symbol = 'USD',
        exchange = 'CCCAGG',
        allData = True,
    ):

    payload = {}
    payload['fsym'] = from_symbol
    payload['tsym'] = to_symbol
    payload['aggregate'] = 1
    payload['e'] = exchange
    payload['allData'] = json.dumps(allData)
    payload['limit'] = 1000
    ret = requests.get(base_url, params=payload).json()
    ret = pd.DataFrame(ret['Data'])

    # additional data
    ret['Date'] = pd.to_datetime(ret['time'], unit='s', utc=True)
    ret['From'] = from_symbol
    ret['To'] = to_symbol
    ret['Exchange'] = exchange
    ret['Volume'] = ret['volumefrom']

    return ret


# Get top coins
# ret = requests.get('https://min-api.cryptocompare.com/data/top/totalvol?fsym=USD&page=0').json()
# ret = pd.DataFrame(ret['Data'])
# top_coins = [ret.iloc[i]['CoinInfo']['Name'] for i in range(len(ret))]
ret = requests.get('https://min-api.cryptocompare.com/data/top/volumes?tsym=USD&limit=20').json()
ret = pd.DataFrame(ret['Data'])
top_coins = ret['SYMBOL'].tolist()
top_coins.remove('XBTUSD')

# Get historical prices for top coins
px_dfs = []

for sym in top_coins:
    print(sym)
    try:
        sym_px = cryptocompare_history(from_symbol = sym, to_symbol = 'USD', exchange = 'CCCAGG')   ## Aggregate price and volume
        sym_px[sym] = sym_px['open']
        sym_px.set_index('Date', inplace=True)
        # restrict to assets with 1+ year histories
        if len(sym_px) > 365:
            px_dfs += [sym_px[[sym]]]
    except:
        print('  Fail')
        pass

comb_px_df = pd.concat(px_dfs, axis=1)
comb_px_df = comb_px_df.loc[:,~comb_px_df.columns.duplicated()]  # drop any duplicate columns (not sure why they are here)



## Correlations ##

# daily
(comb_px_df/comb_px_df.shift(1) - 1.).corr()

# weekly correlation of log pnls
(np.log(comb_px_df/comb_px_df.shift(1))).resample('W').sum().corr()
corr_df = (np.log(comb_px_df/comb_px_df.shift(1))).resample('W').sum().corr()
avg_corr_df = corr_df.copy()
avg_corr_df[avg_corr_df == 1] = np.NaN
avg_corr = avg_corr_df.stack().dropna().mean()


## Volatilities ##

log_rets = np.log(comb_px_df / comb_px_df.shift(1))
# remove some bad data points
log_rets[log_rets.abs() > 0.6] = np.nan

ann_factor = np.sqrt(365.25)    # annualize from daily

# Method 1: rolling window standard deviation of log returns
vol1 = log_rets.rolling(90, min_periods=30, axis=0).std() * ann_factor 
vol1.plot();plt.show()

# Method 2: exponentially weighted to eliminate noise from data moving out of window
vol2 = log_rets.ewm(halflife=45, min_periods=30, axis=0).std() * ann_factor 
vol2.iloc[-365:].plot();plt.show()


# BTC Vol
ax = vol1['BTC'].plot(title='BTC Price Volatility\n(90 day window, annualized)')
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
plt.show()


subset = vol1['BTC'].loc[pd.to_datetime('2015-01-01'):]
(subset.rank()/len(subset)).plot(title='Percentile Rank of 30 Day Vol Since 2015');plt.show()
