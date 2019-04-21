"""
Student Name: Matthew L. Miller
GT User ID: mmiller319
GT ID: 903056227
"""

import pandas as pd
import datetime as dt
from util import get_data
import matplotlib.pyplot as plt
import numpy as np

def get_indicators(df_prices, symbols, window=20):

    # Get prices from the df_prices dataframe, based on symbols.  Then normalize prices.
    prices = df_prices[symbols]
    prices = prices / prices[0]     # Normalize to 1.0

    # Dataframe for the indicators
    df_indicators = pd.DataFrame(index=prices.index)

    #window=20

    """ SMA (Simple Moving Average) - Set as 20 days for now, could be 50, 100, 200..."""
    df_indicators['price'] = prices
    df_indicators['SMA'] = prices.rolling(window=window, center=False).mean()                       # Calc. 20-day SMA
    df_indicators['price_SMA'] = prices / prices.rolling(window=window, center=False).mean()        # Prices/SMA ratio


    """ Bollinger Bands """
    rollingMean = prices.rolling(window=window, center=False).mean()    # Same as SMA
    stddev = prices.rolling(window=window, center=False).std()

    upperBand = rollingMean + 2*stddev
    lowerBand = rollingMean - 2*stddev

    df_indicators['upper band'] = upperBand
    df_indicators['lower band'] = lowerBand

    """ BB Value - Calculated as (prices - rolling mean) / (upper band - lower band)"""
    # bb_value[t] = (price[t] - SMA[t])/(2 * stdev[t])
    bb_value = (prices - rollingMean) / (2 * stddev)
    df_indicators['bb value'] = bb_value


    """ Volatility (Based on 20 days)"""
    volatility = prices.rolling(window=window, center=False).std()
    df_indicators['volatility'] = volatility
    #df_indicators['volatility'] = volatility * 3.5


    """ Momentum (Based on 20 days)"""
    momentum =  prices.diff(window) / prices.shift(window)
    df_indicators['momentum'] = momentum


    """ Relative Strength Index with SMA """
    # Get difference between adj. close price and the adj. close price of the previous day (This indicates
    # whether the stock has gained or lost value
    delta = prices.diff()
    delta = delta[1:]       # delete first row which will be all NaN's

    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0 ] = 0
    down[down > 0] = 0

    # Calculate the EWMA
    rollup1 = pd.stats.moments.ewma(up, 20)
    rolldown1 = pd.stats.moments.ewma(down.abs(), 20)

    # Calculate the RSI based on EWMA
    RS1 = rollup1 / rolldown1
    RSI = 100.0 - (100.0 / (1.0 + RS1))

    # Calculate the SMA
    rollup2 = up.rolling(window=20, center=False).mean()
    rolldown2 = (down.abs()).rolling(window=20, center=False).mean()

    # Calculate the RSI based on SMA
    RS2 = rollup2 / rolldown2
    RSI2 = 100.0 - (100.0 / (1.0 + RS2))

    prices_unnormed = df_prices[symbols]

    df_indicators['prices_unnormed'] = prices_unnormed
    df_indicators['RSI_EMWA'] = RSI
    df_indicators['RSI_SMA'] = RSI2

    return df_indicators



""" Generate plots demonstrating the different indicators according to project specifications:
    Symbol: JPM
    In Sample Period: 1/1/2008 - 12/31/2009
    Out of Sample Period: 1/1/2010 - 12/31/2011
    Starting Cash: $100,000
"""
def test_code():

    startDate = dt.datetime(2008,1,1)
    endDate = dt.datetime(2009,12,31)
    dateRange = pd.date_range(startDate, endDate)
    symbol = 'JPM'

    prices = get_data([symbol], dateRange)
    indicators = get_indicators(prices, symbol)


    """ Graph for SMA """
    sma = indicators[['price', 'SMA', 'price_SMA']]

    fig, ax = plt.subplots()

    ax.plot(sma['price'], label="Price")
    ax.plot(sma['SMA'], label="20-Day SMA", linewidth=2)
    ax.plot(sma['price_SMA'], label="Price/SMA", linewidth=0.85)

    ax.set(xlabel="Jan. 1, 2008 - Dec. 31, 2009", ylabel="Value (Normalized)", title="20-Day Simple Moving Average for JPM")

    ax.set_xlim(startDate, endDate)
    ax.title.set_fontsize(14)
    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    fig.autofmt_xdate()
    plt.show()


    """ Graph for Bollinger Bands"""
    bands = indicators[['price', 'upper band', 'lower band']]

    fig, ax = plt.subplots()
    ax.plot(bands['price'], label="Price")
    ax.plot(bands['upper band'], label="Upper Band", linewidth=0.85, linestyle='dashed')
    ax.plot(bands['lower band'], label="Lower Band", linewidth=0.85, linestyle='dashed')

    ax.set(xlabel="Jan. 1, 2008 - Dec. 31, 2009", ylabel="Value (Normalized)",
           title="Bollinger Bands for JPM Based on 20-Day Moving Average")

    ax.set_xlim(startDate, endDate)
    ax.title.set_fontsize(14)
    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    fig.autofmt_xdate()
    plt.show()

    """ Graph for Bollinger Band Value """
    bb_value = indicators[['price','bb value']]

    fig, ax = plt.subplots()
    #ax.plot(bb_value['price'], label="Price")
    ax.plot(bb_value['bb value'], label="Bollinger Band Value")

    ax.set(xlabel="Jan. 1, 2008 - Dec. 31, 2009", ylabel="Value (Normalized)",
           title="Bollinger Band Value for JPM Based on 20-Day Moving Average")

    ax.set_xlim(startDate, endDate)
    ax.title.set_fontsize(14)
    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)

    ax.axhline(y=0, linewidth=0.85, linestyle='dashed', color='0.5')
    ax.axhline(y=1, linewidth=0.75, linestyle='dashed', color='0.5')
    ax.axhline(y=-1, linewidth=0.75, linestyle='dashed', color='0.5')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    fig.autofmt_xdate()
    plt.show()


    """ Graph for Momentum """
    momentum = indicators[['price','momentum']]

    fig, ax = plt.subplots()
    ax.plot(momentum['price'], label="Price")
    ax.plot(momentum['momentum'], label="momentum")

    ax.set(xlabel="Jan. 1, 2008 - Dec. 31, 2009", ylabel="Stock Price (Normalized)",
           title="Momentum for JPM Over a 20-Day Period")

    ax.set_xlim(startDate, endDate)
    ax.title.set_fontsize(14)
    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    fig.autofmt_xdate()
    plt.show()


    """ Graph for Volatility """
    vol = indicators[['price', 'volatility']]

    fig, ax = plt.subplots()
    ax.plot(vol['price'], label="Price")
    ax.plot(vol['volatility'], label="Volatility")

    ax.set(xlabel="Jan. 1, 2008 - Dec. 31, 2009", ylabel="Value (Normalized)",
           title="Volatility for JPM Based on 20-Day Moving Average")

    ax.set_xlim(startDate, endDate)
    ax.title.set_fontsize(14)
    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)

    y = np.arange(0, 1.3, 0.1)
    plt.yticks(y)
    plt.grid()

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    fig.autofmt_xdate()
    plt.show()

    """ Graph for RSI w/ 20 day rolling mean"""
    rsi = indicators[['price','RSI_SMA', 'RSI_EMWA', 'prices_unnormed']]

    fig, ax = plt.subplots()
    ax.plot(rsi['prices_unnormed'], label="Price")
    ax.plot(rsi['RSI_SMA'], label="RSI Simple Moving Average", linewidth=0.85)
    ax.plot(rsi['RSI_EMWA'], label="RSI Exponential Moving Average", linewidth=0.85)

    ax.set(xlabel="Jan. 1, 2008 - Dec. 31, 2009", ylabel="Value",
           title="RSI for JPM Based on Exponential Moving Average and \n 20-Day Simple Moving Average")

    ax.set_xlim(dt.datetime(2008,1,30), endDate)
    ax.title.set_fontsize(14)
    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)

    y = np.arange(0, 90, 10)
    plt.yticks(y)

    ax.axhline(y=50, linewidth=0.85, linestyle='dashed', color='0.5')
    ax.axhline(y=30, linewidth=0.75, color='0.5')
    ax.axhline(y=70, linewidth=0.75, color='0.5')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    fig.autofmt_xdate()
    plt.show()


if __name__ == "__main__":
    test_code()


def author():
    return 'mmiller319'