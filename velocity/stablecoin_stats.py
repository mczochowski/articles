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
#     'DAI', 
    'GUSD', 
    'PAX', 
    'TUSD', 
    'USDC',
]

decimals = {
#     'DAI': 18,
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


# ### calculate total supply by day

def get_unique_minting_addrs(df):
    df = df[df['from_address'] == root_addr]
    return list(set(df['to_address']))

def get_minted_by_day(df):
    series = df.groupby(df[df['from_address']==root_addr]['block_timestamp'].dt.date)['value'].sum()
    dr = pd.date_range(start=df['block_timestamp'][0].date(), end=df['block_timestamp'].iloc[-1].date())
    series = series.reindex(dr, fill_value=0)
    return series.rename('value_minted')
    

def get_burned_by_day(df):
    series = df.groupby(df[df['to_address']==root_addr]['block_timestamp'].dt.date)['value'].sum()
    dr = pd.date_range(start=df['block_timestamp'][0].date(), end=df['block_timestamp'].iloc[-1].date())
    series = series.reindex(dr, fill_value=0)
    return series.rename('value_burned')


def get_supply_by_day(df):
    minted = get_minted_by_day(df)
    burned = get_burned_by_day(df)
    supply = pd.concat([minted, burned], axis=1).fillna(0).applymap(int)
    supply['supply_EOD'] = supply.cumsum().apply(lambda x: x['value_minted'] - x['value_burned'], axis=1)
    return supply


# ##### DAI has a different issuance mechanism to be calculated later

def get_vol_by_day(df, criterion='txn'):
    assert criterion in ['txn', 'mint', 'burn']
    if criterion == 'txn':
        df = df[(df['from_address'] != root_addr) & (df['to_address'] != root_addr)]
    elif criterion == 'mint':
        df = df[df['from_address'] == root_addr]
    elif criterion == 'burn': 
        df = df[df['to_address'] == root_addr]
    series = df.groupby(df['block_timestamp'].dt.date)['value'].sum()
    series.index = pd.to_datetime(series.index)
    return series.resample('D').asfreq().rename('{}_vol'.format(criterion))


#### timing:

# import time

# df = dfs[symbol]

# print('Method 1: mapping')
# start = time.time()
# df[df.apply(lambda x: x['from_address'] != root_addr and x['to_address'] != root_addr, axis=1)]
# end = time.time()
# print(end - start)


# print('Method 2: built in (vectorized)')
# start = time.time()
# df[(df['from_address'] != root_addr) & (df['to_address'] != root_addr)]
# end = time.time()
# print(end - start)




# # Summary stats for transaction size, mint & burn size

def get_days_in_existence(df):
    return (df['block_timestamp'].max().date() - df['block_timestamp'].min().date())


print('days in existence')
for symbol in symbols:
    print('{}: {}'.format(symbol, get_days_in_existence(dfs[symbol]).days))


def get_txn_size(df, decimals, criterion=''):
    assert criterion in ['', 'excl', 'mint', 'burn']
    if criterion == 'excl':
        df = df[df.apply(lambda x: x['from_address'] != root_addr and x['to_address'] != root_addr, axis=1)]
    elif criterion == 'mint':
        df = df[df['from_address'] == root_addr]
    elif criterion == 'burn':
        df = df[df['to_address'] == root_addr]
    series = (df['value'] / 10. ** decimals).astype(np.float64)
    desc = series.describe()
    ret =desc[['count', 'mean','25%','50%','75%','std']].to_dict()
    ret['sum'] = (series.sum())
    return ret



get_txn_size(dfs['GUSD'], decimals['GUSD'], 'mint')



# pd.options.display.float_format = '{:,.5f}'.format


# ### unique minting addresses:


def get_n_unique_minting_addr(df):
    df = df[df['from_address'] == root_addr]
    return df['to_address'].nunique()


for symbol in symbols:
    print('{}: {}'.format(symbol, get_n_unique_minting_addr(dfs[symbol])))


summary_df = pd.DataFrame(columns=['description','count','mean','25%','50%','75%','std'])
burn_mint_count_ratio = {}
for symbol in symbols:
    incl_desc = get_txn_size(dfs[symbol], decimals[symbol])
    incl_desc['description'] = '{} W/ MINT+BURN'.format(symbol)
    excl_desc = get_txn_size(dfs[symbol], decimals[symbol], 'excl')
    excl_desc['description'] = 'W/O'
    mint_desc = get_txn_size(dfs[symbol], decimals[symbol], 'mint')
    mint_desc['description'] = 'MINT'
    burn_desc = get_txn_size(dfs[symbol], decimals[symbol], 'burn')
    burn_desc['description'] = 'BURN'
    summary_df = summary_df.append(incl_desc, ignore_index=True)
    summary_df = summary_df.append(excl_desc, ignore_index=True)
    summary_df = summary_df.append(mint_desc, ignore_index=True)
    summary_df = summary_df.append(burn_desc, ignore_index=True)
    burn_mint_count_ratio[symbol] = burn_desc['count'] / mint_desc['count']
    print('{} n_minting_addr: {}'.format(symbol, get_n_unique_minting_addr(dfs[symbol])))
    print('{} avg_minting_amount: {}'.format(symbol, mint_desc['sum'] / get_n_unique_minting_addr(dfs[symbol])))
print('Burn-mint count ratio:')
print(burn_mint_count_ratio)
summary_df


# ### distinct addresses

def get_num_distinct_addr(df):
    return len(set(df['from_address']).union(set(df['to_address'])))


for symbol in symbols:
    print('{}: {}'.format(symbol, get_num_distinct_addr(dfs[symbol])))


# ### churn rates

def get_total_minted(df):
    return df[df['from_address'] == root_addr]['value'].sum()
def get_total_burned(df):
    return df[df['to_address'] == root_addr]['value'].sum()

churn_df = pd.DataFrame(index=['Minted', 'Burned', 'Outstanding', 'Churn'], columns=symbols)

for symbol in symbols:
    minted = get_total_minted(dfs[symbol]) / 10. ** decimals[symbol]
    burned = get_total_burned(dfs[symbol]) / 10. ** decimals[symbol]
    total = minted - burned
    burned_ratio = burned / minted
    churn_df[symbol] = [minted, burned, total, burned_ratio]
    print('{}\nMinted: ${}\nBurned: ${}\nCurrent supply: ${}\nBurned ratio (burned / minted): {}'.format(symbol, minted, burned, total, burned_ratio))


# # Time series time

# #### minting activity

mints = pd.concat([get_vol_by_day(dfs[symbol], 'mint')/10**decimals[symbol] for symbol in symbols], axis=1).fillna(0)
mints.columns = symbols


mints.plot(figsize=(20,10))


# #### burning activity

burns = pd.concat([get_vol_by_day(dfs[symbol], 'burn')/10**decimals[symbol] for symbol in symbols], axis=1).fillna(0)
burns.columns = symbols

burns.plot(figsize=(20,10))


# # Frequency of minting / burning / transacting

# #### (Exclude mint & burn)

def get_rate_by_day(df, criterion='txn'):
    df = pd.concat([get_supply_by_day(df), get_vol_by_day(df, criterion)], axis=1)
    series = df['{}_vol'.format(criterion)] / df['supply_EOD']
    return series.rename('{}_rate'.format(criterion))


velocities = pd.concat([get_rate_by_day(dfs[symbol]) for symbol in symbols], axis=1)
velocities.columns = symbols


# #### Exponentially weighted annualized moving average

# velocities.to_csv('{}velocities.csv'.format(data_path))
velocities_cp = velocities.copy()
velocities.loc[1:3, 'TUSD'] = 0
velocities_ewma = velocities.ewm(halflife=45).mean()
(velocities_ewma*365).plot(figsize=(20,10))

velocities_ewma.to_csv('{}velocities_ewma_hl45.csv'.format(data_path))


# #### 30-day annualized rolling average

# (velocities.rolling(30).mean()*365).plot(figsize=(20,10))
(velocities.rolling(30).mean()*365).plot(title='30 Day Rolling Velocities (Annualized)')
plt.show()




# plot dailies
fig, ax = plt.subplots(figsize=(20,10))
ax.plot(velocities)
ax.legend(symbols)
plt.show()


# ### minting


mint_rate = pd.concat([get_rate_by_day(dfs[symbol], 'mint') for symbol in symbols], axis=1).fillna(0)
mint_rate.columns = symbols


sns.set()
fig, ax = plt.subplots(figsize=(20,10))
ax.plot(mint_rate)
ax.legend(symbols)
plt.show()


# ### burning

burn_rate = pd.concat([get_rate_by_day(dfs[symbol], 'burn') for symbol in symbols], axis=1).fillna(0)
burn_rate.columns = symbols

sns.set()
fig, ax = plt.subplots(figsize=(20,10))
ax.plot(burn_rate)
ax.legend(symbols)
plt.show()


#### MCZ Edits #####

txs = pd.concat(dfs, axis=0)
txs[['transaction_hash']].count(axis=0, level=0)

txs['token'] = txs.index.get_level_values(0)
txs['decimals'] = txs.index.get_level_values(0).map(lambda x: decimals[x])
txs['dollar_value'] = txs['value'] / np.power(10,txs['decimals'])


# txs.loc['GUSD'].groupby('to_address')['value'].sum().subtract(txs.loc['GUSD'].groupby('from_address')['value'].sum(),fill_value=0)

# # pd.concat([txs.loc['GUSD'].groupby('to_address')['value'].sum(), txs.loc['GUSD'].groupby('from_address')['value'].sum()], axis=1, names=['to','from'])
# addr = pd.DataFrame({
#                      'to': txs.loc['GUSD'].groupby('to_address')['value'].sum(), 
#                      'from': txs.loc['GUSD'].groupby('from_address')['value'].sum(),
#                      'min_block': txs.groupby('to_address')['block_number'].min(),
#                      'max_block': txs.groupby('from_address')['block_number'].max()
#                     })
# addr['bal'] = addr['to'] - addr['from'].fillna(0)
# addr['block_diff'] = addr['max_block'] - addr['min_block']

# # temporary addresses
# addr[(addr['bal'] == 0) & (addr['block_diff'] < 100)]


# gusd = txs.loc['GUSD']

# # exclude mints
# # gusd[gusd['from_address']==root_addr]['value'].sum()
# gusd[gusd['from_address']==root_addr]

# # exclude burns
# gusd[gusd['to_address']==root_addr]

# # excludes send to self:
# txs[txs['to_address'] == txs['from_address']].head()


# # exclude origination addresses
# orig_addr = gusd[gusd['from_address']==root_addr]['to_address']
# # gusd[gusd['from_address']==root_addr].groupby('to_address')['transaction_hash'].count()
# second_order = gusd[gusd['from_address'].isin(gusd[gusd['from_address']==root_addr]['to_address'].unique())]
# # len(second_order['to_address'].unique())



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
addr.dropna(axis=0, subset=['bal']).groupby(level=1)['bal'].sum()
addr.dropna(axis=0, subset=['bal']).groupby(level=1)['bal'].sum().sort_values(ascending=False)
# Sources:
# https://api.tokenanalyst.io/#Address
# https://etherscan.io/accounts/1?&l=Exchange

#### End Aside ####

# exchange addresses
ex_addrs = pd.read_csv('exchange_addresses.csv')
addrs_by_bal = addr.dropna(axis=0, subset=['bal']).groupby(level=1)['bal'].sum().sort_values(ascending=False)
pd.merge(left=pd.DataFrame(addrs_by_bal), right=ex_addrs[['Address', 'Name']], left_index=True, right_on='Address', how='left')

######
# velocity excluding self-churn #
txs_v = txs.copy()

# minting & burning
txs_v = txs_v[(txs_v['to_address'] != root_addr) & (txs_v['from_address'] != root_addr)]    

# self sends
txs_v = txs_v[txs_v['to_address'] != txs_v['from_address']] 
# addr[(addr['bal'] == 0) & (addr['block_diff'] < 100)]     # short life of address; not very large

# PAX and USDC (and possibly the others) mint to an intermediate address
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

# remove transactions involving known exchanges
txs_v = txs_v[(~txs_v['to_address'].isin(ex_addrs['Address'])) & (~txs_v['from_address'].isin(ex_addrs['Address']))]

# remove temporary addresses: current balance is 0 and had non-zero balance for less than 60 blocks
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


