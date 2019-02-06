import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import requests
import json as json

sns.set()


# # Preprocessing

# ### Load data

root_addr = '0x0000000000000000000000000000000000000000'

symbols = [
    'GUSD', 
    'PAX', 
    'TUSD', 
    'USDC',
]

decimals = {
    'GUSD': 2,
    'PAX': 18,
    'TUSD': 18,
    'USDC': 6,
}

data_path = './dataset/'
dfs = {symbol: pd.read_csv('{}{}.csv'.format(data_path, symbol), 
                           converters={'value':int}, 
                           parse_dates=['block_timestamp']) 
       for symbol in symbols}



txs = pd.concat(dfs, axis=0)
txs[['transaction_hash']].count(axis=0, level=0)

txs['token'] = txs.index.get_level_values(0)
txs['decimals'] = txs.index.get_level_values(0).map(lambda x: decimals[x])
txs['dollar_value'] = txs['value'] / np.power(10,txs['decimals'])


###########################
# VELOCITY AND SELF-CHURN #

# number of mints:
txs[txs['from_address']==root_addr]['to_address'].groupby(level=0).count()

# Get address statistics
addr = pd.DataFrame({
                     'to': txs.groupby(by=['token','to_address'])['dollar_value'].sum(), 
                     'from': txs.groupby(by=['token','from_address'])['dollar_value'].sum(),
                     'min_block': txs.groupby(by=['token','to_address'])['block_number'].min(),
                     'max_block': txs.groupby(by=['token','from_address'])['block_number'].max()
                    })
addr['bal'] = addr['to'] - addr['from'].fillna(0)
addr['block_diff'] = addr['max_block'] - addr['min_block']



#### Aside: ####

# NaN Addresses:
addr[addr['bal'].isnull()].groupby(level=0).count()
nan_addr = addr[addr['bal'].isnull()].index.get_level_values(1)
nan_txs = txs[(txs['from_address'].isin(nan_addr)) | (txs['to_address'].isin(nan_addr))]

# usually due to a send "from" an address with no previous balance
nax = '0x41a05cc43654a2814f83f29a7d6bb27c418111e8'
txs[(txs['to_address'] == nax) | (txs['from_address'] == nax)].loc['USDC']
addr.xs(nax,level=1)

# get largest addresses:
# Sources:
# https://api.tokenanalyst.io/#Address
# https://etherscan.io/accounts/1?&l=Exchange
# addr.dropna(axis=0, subset=['bal']).groupby(level=1)['bal'].sum().sort_values(ascending=False)

#### End Aside ####



# exchange addresses
ex_addrs = pd.read_csv('exchange_addresses.csv')
addrs_by_bal = addr.dropna(axis=0, subset=['bal']).groupby(level=1)['bal'].sum().sort_values(ascending=False)
pd.merge(left=pd.DataFrame(addrs_by_bal), right=ex_addrs[['Address', 'Name']], left_index=True, right_on='Address', how='left')

######
# velocity excluding self-churn #
txs_v = txs.copy()

# 1. minting & burning
txs_v = txs_v[(txs_v['to_address'] != root_addr) & (txs_v['from_address'] != root_addr)]    

# 2. self sends
txs_v = txs_v[txs_v['to_address'] != txs_v['from_address']] 
# addr[(addr['bal'] == 0) & (addr['block_diff'] < 100)]     # short life of address; not very large

# 3. PAX and USDC (and possibly the others) mint and burn to an intermediate address
# second_order = gusd[gusd['from_address'].isin(gusd[gusd['from_address']==root_addr]['to_address'].unique())] 
# number of unique addresses: txs[txs['from_address']==root_addr]['to_address'].groupby(level=0).nunique()
issued_addrs = txs[txs['from_address']==root_addr]['to_address'].groupby(level=0).unique() 
second_order_orig = ['PAX', 'USDC']
so_addr = np.concatenate([issued_addrs[so_token] for so_token in second_order_orig])
txs_v = txs_v[~txs_v['from_address'].isin(so_addr)]     # may overcount if another token for some reason sends to their 2nd order origination address

redeem_addrs = txs[txs['to_address']==root_addr]['from_address'].groupby(level=0).unique()
second_order_burn = ['PAX', 'USDC', 'GUSD']
sb_addr = np.concatenate([redeem_addrs[sb_token] for sb_token in second_order_burn])
txs_v = txs_v[~txs_v['to_address'].isin(sb_addr)]     # may overcount if another token for some reason sends to their 2nd order burn address

# 4. remove transactions involving known exchanges
txs_v = txs_v[(~txs_v['to_address'].isin(ex_addrs['Address'])) & (~txs_v['from_address'].isin(ex_addrs['Address']))]

# 5. remove temporary addresses: current balance is 0 and had non-zero balance for less than 60 blocks
temp_addr = addr[(addr['bal'] == 0) & (addr['block_diff'] < 60)]
# temp_addr.reset_index().drop_duplicates(subset=['level_1','level_0'])
temp_addr_list = temp_addr.index.get_level_values(1).unique()
txs_v = txs_v[(~txs_v['to_address'].isin(temp_addr_list)) & (~txs_v['from_address'].isin(temp_addr_list))]



# adjusted dollar volume
adj_vols_dly = txs_v.groupby(by=['token', 'TxDate'])['dollar_value'].sum()

# get daily supply
minted_dly = txs[txs['from_address']==root_addr].groupby(by=['token', 'TxDate'])['dollar_value'].sum()
burned_dly = txs[txs['to_address']==root_addr].groupby(by=['token', 'TxDate'])['dollar_value'].sum()
net_chg_dly = minted_dly.subtract(burned_dly, fill_value=0)
supply_dly = net_chg_dly.unstack(level=0).cumsum().fillna(method='ffill')
# supply_dly.iloc[-1] == churn_df.loc['Outstanding']

velocity_dly = adj_vols_dly.unstack(level=0) / supply_dly

# if we already have an observation (i.e. token exists), then NaNs should be 0 (i.e. no volume on that day)
for i in range(1,len(velocity_dly)):
    prev_row = velocity_dly.iloc[i-1]
    row = velocity_dly.iloc[i]
    row[(~prev_row.isnull()) & (row.isnull())] = 0
    velocity_dly.iloc[i] = row

# plot 30 day moving average velocity without churn
(velocity_dly.rolling(window=30, min_periods=10).mean()*365).plot(title='30 Day Rolling Velocities (Annualized)\nSelf-Churn Removed')
plt.show()

# without removing self-churn
(velocities.rolling(30).mean()*365).plot(title='30 Day Rolling Velocities (Annualized)\nSelf-Churn NOT Removed')
plt.show()
