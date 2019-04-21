"""
Student Name: Matthew L. Miller
GT User ID: mmiller319
GT ID: 903056227
"""

import datetime as dt
import pandas as pd
import util as ut
import random

import RTLearner as rt
import BagLearner as bl
from indicators import *


class StrategyLearner(object):

    # constructor
    def __init__(self, verbose=False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        self.N = 5
        self.learner = bl.BagLearner(learner=rt.RTLearner, kwargs={'leaf_size': 5}, bags=20, boost=False, verbose=False)

    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):

        """ Get prices and indicator data """
        dateRange = pd.date_range(sd, ed)
        df_prices = get_data([symbol], dateRange)
        prices = df_prices[symbol]
        prices = prices / prices[0]  # Normalize to 1.0
        prices = prices.to_frame()  # prices becomes a series when normalize, convert back to a dataframe

        # Get data for SPY as a benchmark
        pricesSPY = df_prices['SPY']
        pricesSPY = pricesSPY / pricesSPY[0]  # Normalize to 1.0

        # Get indicators for the stock (SMA, Bollinger Bands, Volatility and RSI)
        lookback = 20
        indicators = get_indicators(prices, symbol, window=lookback)    # Add lookback period for indicator's window

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

        # Create dataframe that holds the selected indicators to be used in training model
        """ Indicators used in Manual Strategy Project: RSI_SMA and BB_Value"""
        trainingIndicators = pd.concat((RSI_SMA, BB_value), axis=1)
        trainingIndicators.fillna(0, inplace=True)
        trainingIndicators = trainingIndicators[:-self.N]       # Eliminate last N days to account for N-day difference

        dataX = trainingIndicators.values       # Final training dataset holding values for indicators

        """ --------------------------------------------------------------------------------------------- 
        Classify the positions according to the following:
            1 = LONG
            0 = CASH
            -1 = SHORT        

        Whether to buy, sell, or hold is determined by the N-day return: If it is above 0.02 + impact,
        we buy, if less than 0.02 - impact we sell, and otherwise hold.   
        N is currently set to 5.   """

        numTradingDays = prices.shape[0]

        dataY = np.empty(numTradingDays - self.N)  # Hold labels 1,-1,0 corresponding to LONG,SHORT,CASH

        # Thresholds where model learns to buy or sell
        # Test 0: ML4T-220 will fail if these are set to more than +/- 0.4 + impact
        YBUY = 0.04 + self.impact
        YSELL = -(0.04 + self.impact)

        # Cycle through prices data, append 1 or-1 when price difference over N days exceeds
        # BUY/SELL thresholds and 0 otherwise
        for i in range(0, numTradingDays - self.N):

            # Calculate N-day return
            # price_diff = prices.ix[i+self.N, symbol] - prices.ix[i, symbol]
            # N_return = price_diff / prices.ix[i, symbol]
            N_return = (prices.ix[i + self.N, symbol] / prices.ix[i, symbol]) - 1

            if N_return > YBUY:
                dataY[i] = 1  # LONG Position
            elif N_return < YSELL:
                dataY[i] = -1  # SHORT position
            else:
                dataY[i] = 0  # CASH position

        # Convert dataY to np.array and pass dataX and dataY to the learner
        # Learner will create model according to indicator data (dataX) and the associated BUY/SELL/HOLD
        # labels (dataY)
        dataY = np.array(dataY)
        self.learner.addEvidence(dataX, dataY)

    """ End addEvidence() """
    """---------------------------------------------------------------------------------------------"""

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol="IBM", \
                   sd=dt.datetime(2009, 1, 1), \
                   ed=dt.datetime(2010, 1, 1), \
                   sv=10000):

        # here we build a fake set of trades
        # your code should return the same sort of data
        # dates = pd.date_range(sd, ed)
        # prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        # trades = prices_all[[symbol,]]  # only portfolio symbols
        # trades_SPY = prices_all['SPY']  # only SPY, for comparison later
        # trades.values[:,:] = 0 # set them all to nothing
        # trades.values[0,:] = 1000 # add a BUY at the start
        # trades.values[40,:] = -1000 # add a SELL
        # trades.values[41,:] = 1000 # add a BUY
        # trades.values[60,:] = -2000 # go short from long
        # trades.values[61,:] = 2000 # go long from short
        # trades.values[-1,:] = -1000 #exit on the last day
        # if self.verbose: print type(trades) # it better be a DataFrame!
        # if self.verbose: print trades
        # if self.verbose: print prices_all
        # return trades

        """ Get prices and indicator data"""
        dateRange = pd.date_range(sd, ed)
        df_prices = get_data([symbol], dateRange)
        prices = df_prices[symbol]
        prices = prices / prices[0]  # Normalize to 1.0
        prices = prices.to_frame()  # prices becomes a series when normalize, convert back to a dataframe

        # Get data for SPY as a benchmark
        pricesSPY = df_prices['SPY']
        pricesSPY = pricesSPY / pricesSPY[0]  # Normalize to 1.0

        # Get indicators for the stock (SMA, Bollinger Bands, Volatility and RSI)
        lookback = 20
        indicators = get_indicators(prices, symbol, window=lookback)  # Add lookback period for indicator's window

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

        # Create testing indicators dataframe that holds the same indicators used in the training model
        # Indicators are RSI_SMA, BB_Value and Volatility - same as in Manual Strategy Project
        testingIndicators = pd.concat((RSI_SMA, BB_value, volatility), axis=1)
        testingIndicators.fillna(0, inplace=True)

        # Testing data to be passed to the model in order to predict BUY/SELL/HOLD labels
        testX = testingIndicators.values

        # Get predicted labels from the learner - testY will be a series of labels predicted whether to
        # be LONG, SHORT, or CASH
        testY = self.learner.query(testX)

        """ Cycle through testY dataframe, BUY and SELL stock based on conditions for indicators
            (Similar to Manual Strategy project)
        """
        numDates = testY.shape[0] - 1
        holdings = 0

        orders = prices.copy()
        orders[:] = "BUY"
        orders.columns = ['Order']  # holds type of order (BUY or SELL)

        shares = prices.copy()
        shares[:] = 0
        shares.columns = ['Shares']  # number of shares bought/sold in each order

        symbols = prices.copy()
        symbols[:] = symbol
        symbols.columns = ['Symbol']  # Symbol of stock being traded

        # BUY, SELL, or HOLD depending on whether last trade was a buy or a sell
        # Set to HOLD for first day of trading
        lastPosition = 0

        for i in range(0, numDates - 1):

            # if we're currently in CASH, we can either go LONG or SHORT
            if lastPosition == 0 and holdings == 0:

                # LONG 1000 shares
                if testY[i] > 0:
                    shares.values[i, :] = 1000
                    orders.values[i] = "BUY"
                    holdings += 1000  # net holdings = +1000
                    lastPosition = 1
                # SHORT 1000 shares
                elif testY[i] < 0:
                    shares.values[i, :] = 1000
                    orders.values[i] = "SELL"
                    holdings -= 1000  # net holdings = -1000
                    lastPosition = -1
                # Remain in CASH
                elif testY[i] == 0:  # net holdings = 0
                    shares.values[i, :] = 0
                    orders.values[i] = "BUY"
                    lastPosition = 0

            # if we're currently LONG 1000 shares, we can SHORT 2000 shares or go to CASH
            elif lastPosition == 1 and holdings == 1000:

                # SHORT 2000 shares
                if testY[i] <= 0:
                    shares.values[i, :] = 2000
                    orders.values[i] = "SELL"
                    holdings -= 2000  # net holdings = -1000
                    lastPosition = -1
                # Convert to CASH
                elif testY[i] == 0:
                    shares.values[i, :] = 1000
                    orders.values[i] = "SELL"
                    holdings -= 1000  # net holdings = 0
                    lastAction = 0


            # if we're currently SHORT, we can go LONG 2000 shares or go to CASH
            elif lastPosition == -1 and holdings == -1000:

                # LONG 2000 shares
                if testY[i] >= 0:
                    shares.values[i, :] = 2000
                    orders.values[i] = "BUY"
                    holdings += 2000  # net holdings of +1000
                    lastPosition = 1
                # Convert to CASH
                elif testY[i] == 0:
                    shares.values[i, :] = 1000
                    orders.values[i] = "BUY"
                    holdings += 1000  # net holdings of 0
                    lastPosition = 0


        # Action for final day of trading in the given time period
        # If SHORT, buy 1000 shares, if LONG, sell 1000 shares
        if lastPosition == -1 and holdings == -1000:
            shares.values[numDates - 1, :] = 1000
            orders.values[i] = "BUY"
            holdings += 1000  # Net holdings = 0

        elif lastPosition == 1 and holdings == 1000:
            shares.values[numDates - 1, :] = 1000
            orders.values[i] = "SELL"
            holdings -= 1000  # Net holdings = 0


        # for i in range(orders.shape[0]):
        #     print(orders[i])
        # print("orders = ")
        # print(orders)

        results = pd.concat((symbols, orders, shares), axis=1)

        return results


    def author(self):
        return 'mmiller319'


if __name__ == "__main__":
        sl = StrategyLearner()
        sl.addEvidence()
        sl.testPolicy()
        print "One does not simply think up a strategy"









