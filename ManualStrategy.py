import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data
import matplotlib.pyplot as plt
from marketsimcode import compute_portvals
from indicators import get_indicators
import experiment1 as exp1
import StrategyLearner as sl


def testPolicy(symbol=['AAPL'], sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):

    dateRange = pd.date_range(sd, ed)
    df_prices = get_data([symbol], dateRange)
    prices = df_prices[symbol]
    prices = prices / prices[0]         # Normalize to 1.0
    prices = prices.to_frame()          # prices becomes a series when normalize, convert back to a dataframe

    # Get data for SPY as a benchmark
    pricesSPY = df_prices['SPY']
    pricesSPY = pricesSPY / pricesSPY[0]     # Normalize to 1.0

    # Set rolling window size
    rollingWindow = 20

    # Get indicators for the stock (SMA, Bollinger Bands, Volatility and RSI)
    indicators = get_indicators(prices, symbol)

    # Indicators - not all of these will necessarily be needed.
    sma = indicators['SMA']
    price_SMA = indicators['price_SMA']
    BB_upper = indicators['upper band']
    BB_lower = indicators['lower band']
    BB_value = indicators['bb value']
    volatility = indicators['volatility']
    momentum = indicators['momentum']
    RSI_EMWA = indicators['RSI_EMWA']
    RSI_SMA = indicators['RSI_SMA']


    """ Cycle through prices dataframe, BUY or SELL stock based on conditions for indicators
    """
    numDates = prices.shape[0]
    holdings = 0

    orders = prices.copy()
    orders.columns=['Order']        # holds type of order (BUY or SELL)
    orders[:] = ''

    shares = prices.copy()
    shares.columns = ['Shares']     # number of shares bought/sold in each order
    shares[:] = 0

    symbols = prices.copy()
    symbols.columns = ['Symbol']    # Symbol of stock being traded
    symbols[:] = symbol


    RSI_SMA_top = 60
    RSI_SMA_bottom = 40

    momentum_top = 0.25
    momentum_bottom = -0.25

    volatility_top = 0.25
    volatility_bottom = 0.15

    BB_value_top = 0.25
    BB_value_bottom = -0.25

    for i, row in prices.iterrows():

        # Get prices for current index
        currentSMA = sma.loc[i]
        currentPrice_SMA = price_SMA.loc[i]
        currentBB_value = BB_value.loc[i]
        currentMomentum = momentum.loc[i]
        currentVolatility = volatility.loc[i]
        currentRSI_SMA = RSI_SMA.loc[i]
        currentPrice = row[symbol]


        if (currentRSI_SMA > RSI_SMA_top) and (currentBB_value>BB_value_top) and (holdings<1000):

            orders.loc[i]['Order'] = 'BUY'

            if holdings == 0:
                shares.loc[i]['Shares'] = 1000
                holdings += 1000
            else:
                shares.loc[i]['Shares'] = 2000
                holdings += 2000

        elif (currentRSI_SMA < RSI_SMA_bottom) and (currentBB_value<BB_value_bottom) and (holdings>-1000):

            orders.loc[i]['Order'] = 'SELL'

            if holdings == 0:
                shares.loc[i]['Shares'] = 1000
                holdings -= 1000
            else:
                shares.loc[i]['Shares'] = 2000
                holdings -= 2000



    trades = pd.concat([symbols, orders, shares], axis=1)
    trades.columns = ['Symbol', 'Order', 'Shares']
    trades = trades[trades.Shares != 0]

    return trades






""" Test code for Manual Strategy with the following parameters:
    symbol:  Stock symbol, in this case JPM
    startDate: 1/1/2008
    endDate: 12/31/2009
    starting value (sv) = $100,000
"""
def test_code():

    # In sample period: 1/1/2008 - 12/31/2009
    startDate = dt.datetime(2008, 1, 1)
    endDate = dt.datetime(2009, 12, 31)

    # Out of sample period:  1/1/2010 - 12/31/2011
    # startDate = dt.datetime(2010, 1, 1)
    # endDate = dt.datetime(2011, 12, 31)

    dateRange = pd.date_range(startDate, endDate)
    symbol = 'JPM'

    df_prices = get_data([symbol], dateRange)
    prices = df_prices[symbol]                  # Get prices for JPM
    prices_SPY = df_prices['SPY']               # Get prices for SPY to be used as comparison
    prices = prices / prices[0]                 # Normalize to 1.0
    prices_SPY = prices_SPY / prices_SPY[0]

    # Starting value of portfolio
    sv = 100000

    # Generate results of testPolicy for JPM with $100,000 between 1/1/2008 and 12/31/2009
    results = testPolicy(symbol="JPM", sd=startDate, ed=endDate, sv=sv)

    # print("results =")
    # print(results)

    # Get portfolio values using the compute_portvals function from marketsimcode.py
    portvals = compute_portvals(results, sv, 0, 0)
    portvals = portvals / portvals[0]

    """-----------------------------------------------------------------------------------------"""
    # Use Strategy Learner to get values for same time period
    learner = exp1.StrategyLearner(False, 0.0)

    learner.addEvidence(symbol="JPM", sd=startDate, ed=endDate, sv=sv)

    strategy_results = learner.testPolicy(symbol="JPM", sd=startDate, ed=endDate, sv=sv)

    strategy_portvals = compute_portvals(strategy_results, sv, 0, 0)
    strategy_portvals = strategy_portvals / strategy_portvals[0]
    """-----------------------------------------------------------------------------------------"""

    """ Generate graph of Manual Strategy compared to performance of SPY """
    fig, ax = plt.subplots()

    ax.plot(portvals, 'k', label="Manual Strategy")
    ax.plot(prices_SPY, 'b', label="SPY")
    ax.plot(strategy_portvals, 'g', label="Strategy Learner")

    ax.set(xlabel="Jan. 1, 2008 - Dec. 31, 2009", ylabel="Value (Normalized)",
           title="Comparison of Portfolio Performance During In-Sample Period \n between Manual Strategy, Strategy Learner, and SPY")

    ax.set_xlim(startDate, endDate)
    ax.title.set_fontsize(14)
    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    fig.autofmt_xdate()
    plt.grid()
    #plt.show()
    fig.savefig('experiment1_insample.png')
    plt.close(fig)


    """ Repeat Graph for Out-of-Sample Period """
    """ ----------------------------------------------------------------------------------------- """
    # Out of sample period:  1/1/2010 - 12/31/2011
    startDate = dt.datetime(2010, 1, 1)
    endDate = dt.datetime(2011, 12, 31)

    dateRange = pd.date_range(startDate, endDate)
    symbol = 'JPM'

    df_prices = get_data([symbol], dateRange)
    prices = df_prices[symbol]                  # Get prices for JPM
    prices_SPY = df_prices['SPY']               # Get prices for SPY to be used as comparison
    prices = prices / prices[0]                 # Normalize to 1.0
    prices_SPY = prices_SPY / prices_SPY[0]

    # Starting value of portfolio
    sv = 100000

    # Generate results of testPolicy for JPM with $100,000 between 1/1/2008 and 12/31/2009
    results = testPolicy(symbol="JPM", sd=startDate, ed=endDate, sv=sv)

    # Get portfolio values using the compute_portvals function from marketsimcode.py
    portvals = compute_portvals(results, sv, 0, 0)
    portvals = portvals / portvals[0]

    """-----------------------------------------------------------------------------------------"""
    # Use Strategy Learner to get values for same time period
    learner = exp1.StrategyLearner(False, 0.0)

    learner.addEvidence(symbol="JPM", sd=startDate, ed=endDate, sv=sv)

    strategy_results = learner.testPolicy(symbol="JPM", sd=startDate, ed=endDate, sv=sv)

    strategy_portvals = compute_portvals(strategy_results, sv, 0, 0)
    strategy_portvals = strategy_portvals / strategy_portvals[0]
    """-----------------------------------------------------------------------------------------"""

    """ Generate graph of Manual Strategy & Strategy Learner compared to performance of SPY
        for out-of-sample period """
    fig, ax = plt.subplots()

    ax.plot(portvals, 'k', label="Manual Strategy")
    ax.plot(prices_SPY, 'b', label="SPY")
    ax.plot(strategy_portvals, 'g', label="Strategy Learner")

    ax.set(xlabel="Jan. 1, 2010 - Dec. 31, 2011", ylabel="Value (Normalized)",
           title="Comparison of Portfolio Performance During Out-of-Sample Period \n between Manual Strategy, Strategy Learner, and SPY")

    ax.set_xlim(startDate, endDate)
    ax.title.set_fontsize(14)
    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    fig.autofmt_xdate()
    plt.grid()
    #plt.show()
    fig.savefig('experiment1_outofsample.png')
    plt.close(fig)


if __name__ == "__main__":
    test_code()


