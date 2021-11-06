#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 19:49:29 2021

@author: oem
"""
#%%
#%% ##################################
#%% portfolio allocation 
### https://pyportfolioopt.readthedocs.io/en/latest/
### https://github.com/robertmartin8/PyPortfolioOpt/blob/master/cookbook/4-Black-Litterman-Allocation.ipynb
###############################################################################
###############################################################################

#%% get the initial imports 
# Initial imports
import os
import requests
import pandas as pd
#from dotenv import load_dotenv
#import alpaca_trade_api as tradeapi
import json
import numpy as np
from fbprophet import Prophet
from scipy.optimize import linprog
from cvxpy import *
import yfinance as yf
import scipy as sp
import matplotlib.pyplot as plt

import datetime 
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

import pypfopt
pypfopt.__version__

# calc risk models -- covariance shrinkage 

from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting

#%% set up the directory
os.chdir("/home/oem/Fintech_0/Project_3") 

#%matplotlib inline
#%% load the keys
load_dotenv()

alpaca_api_key = os.getenv("ALPACA_API_KEY")
alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")

print(f"Alpaca Key type: {type(alpaca_api_key)}")
print(f"Alpaca Secret Key type: {type(alpaca_secret_key)}")


#%%  get data from alpaca
# Set Alpaca API key and secret
# create the Alpaca API object 

alpaca = tradeapi.REST(
    alpaca_api_key,
    alpaca_secret_key, 
    api_version='v2')

# Format current date as ISO format -- make sure it is a weekday
end1 = pd.Timestamp("2021-11-05", tz="America/New_York").isoformat()
start1 = pd.Timestamp("2018-11-05", tz="America/New_York").isoformat()

tickers = ["AAPL","TSLA","GOOG", "MSFT", "AMZN", "NAT", "BAC", "DPZ", "DIS", "KO", "MCD", "COST", "SBUX"]


# set the time frame
timeframe = "1D"

## get the current closing prices and convert to a dataframe 
df_portfolio = alpaca.get_barset(
    tickers,
    timeframe, 
    start = start1, 
    end = end1,
    limit =1000
).df

#df_portfolio.index = pd.to_datetime(df_portfolio.index,format="%Y-%m-%d")

print(df_portfolio)

df_portfolio.to_csv("data_stock_003.csv", index=None)
df_portfolio.to_csv("data_stock_indx.csv")
dff=df_portfolio


#%% get the ohlc prices 
ohlc = yf.download(tickers, period="max")
prices = ohlc["Adj Close"]
prices.tail()

df = pd.DataFrame(ohlc)
df.to_csv('stock_data_ohlc.csv')

#%% get the SPY

market_prices = yf.download("SPY", period="max")["Adj Close"]
market_prices.head()
#%%  calc market cap 

mcaps = {}
for t in tickers:
    stock = yf.Ticker(t)
    mcaps[t] = stock.info["marketCap"]
mcaps
#%%  load in port optimization 


S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
delta = black_litterman.market_implied_risk_aversion(market_prices)
delta
#%% plot the covariance 
plotting.plot_covariance(S, plot_correlation=True);


#%% get the prior returns 
market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)
market_prior
#%%
market_prior.plot.barh(figsize=(10,5));
#%%


viewdict = {
    "AAPL": 0.10, 
    "TSLA": 0.05,
    "GOOG": 0.05,
    "AMZN": 0.10,
    "BAC": 0.30,
    "COST": 0.05,
    "DIS": 0.05,
    "DPZ": 0.20,
    "KO": -0.05,  # I think Coca-Cola will go down 5%
    "MCD": 0.10,
    "MSFT": 0.05,
    "NAT": 0.05,  # but low confidence, which will be reflected later
    "SBUX": 0.05
}

bl = BlackLittermanModel(S, pi=market_prior, absolute_views=viewdict)
#%%

confidences = [
    0.4,
    0.7,
    0.8,
    0.6,
    0.4,
    0.2,
    0.5,
    0.7, # confident in dominos
    0.7, # confident KO will do poorly
    0.7, 
    0.5,
    0.1,
    0.4
]

#%% BL models 
bl = BlackLittermanModel(S, pi=market_prior, absolute_views=viewdict, omega="idzorek", view_confidences=confidences)


#%% plots 


fig, ax = plt.subplots(figsize=(7,7))
im = ax.imshow(bl.omega)

# We want to show all ticks...
ax.set_xticks(np.arange(len(bl.tickers)))
ax.set_yticks(np.arange(len(bl.tickers)))

ax.set_xticklabels(bl.tickers)
ax.set_yticklabels(bl.tickers)
plt.show()

#%%

np.diag(bl.omega)
#%%  compute the variances 


intervals = [
    (0, 0.25),
    (0.1, 0.4),
    (-0.1, 0.15),
    (-0.05, 0.1),
    (0.15, 0.25),
    (-0.1, 0),
    (0.1, 0.2),
    (0.08, 0.12),
    (0.1, 0.9),
    (0, 0.3)
]


variances = []
for lb, ub in intervals:
    sigma = (ub - lb)/2
    variances.append(sigma ** 2)

print(variances)
omega = np.diag(variances)


#%% BL measures 

bl = BlackLittermanModel(S, pi="market", market_caps=mcaps, risk_aversion=delta,
                        absolute_views=viewdict, omega=omega)

#%% get bl returns 


# Posterior estimate of returns
ret_bl = bl.bl_returns()
ret_bl
#%%  get the returns for market priors 

rets_df = pd.DataFrame([market_prior, ret_bl, pd.Series(viewdict)], 
             index=["Prior", "Posterior", "Views"]).T
rets_df
#%%

rets_df.plot.bar(figsize=(12,8));

#%%

S_bl = bl.bl_cov()
plotting.plot_covariance(S_bl);
#%%


from pypfopt import EfficientFrontier, objective_functions

#%% efficient frontier

ef = EfficientFrontier(ret_bl, S_bl)
ef.add_objective(objective_functions.L2_reg)
ef.max_sharpe()
weights = ef.clean_weights()
weights

ef = EfficientFrontier(0.7, S, weight_bounds=(None, None))
ef.add_constraint(lambda w: w[0] >= 0.2)
ef.add_constraint(lambda w: w[2] == 0.15)
ef.add_constraint(lambda w: w[3] + w[4] <= 0.10)

fig, ax = plt.subplots()
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)
plt.show()
#%%

pd.Series(weights).plot.pie(figsize=(10,10));
#%%


from pypfopt import DiscreteAllocation

da = DiscreteAllocation(weights, prices.iloc[-1], total_portfolio_value=20000)
#alloc, leftover = da.lp_portfolio()
#print(f"Leftover: ${leftover:.2f}")
#alloc

#%%

print(da.weights)

#%%
pd.Series(da.weights).plot.pie(figsize=(10,10));

#%%
#%% eff frontier

ef = EfficientFrontier(mu, S, weight_bounds=(None, None))
ef.add_constraint(lambda w: w[0] >= 0.2)
ef.add_constraint(lambda w: w[2] == 0.15)
ef.add_constraint(lambda w: w[3] + w[4] <= 0.10)

fig, ax = plt.subplots()
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)
plt.show()

#%%%















































