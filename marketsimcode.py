import pandas as pd  		   	  			    		  		  		    	 		 		   		 		  
import numpy as np  		   	  			    		  		  		    	 		 		   		 		  
import datetime as dt  		   	  			    		  		  		    	 		 		   		 		  
import os  		   	  			    		  		  		    	 		 		   		 		  
from util import get_data, plot_data

""" Instead of an orders file, the parameter 'orders' is a dataframe passed to compute_portvals when called
    from another file
"""
def compute_portvals(orders, start_val = 1000000, commission=0.00, impact=0.00):

    # Sort orders by index
    orders = orders.sort_index()

    # Crete a list of all symbols in the orders
    symbols = orders['Symbol'].values
    symbolsList = list(set(symbols))

    """ orders should be indexed by date already"""
    # # Convert 'Date' column to datetime objects
    # orders.index = pd.to_datetime(orders.index)

    # Get start date and end date and create a date range from start to end
    startDate = orders.index.values[0]
    endDate = orders.index.values[-1]
    dateRange = pd.date_range(startDate, endDate)

    # Create a dataframe holding the price for each symbol and add column to hold the amount of cash
    # available (will be all ones at first)
    prices = get_data(symbolsList, dateRange)
    prices['Cash'] = 1.00

    # Create dataframes for trades & holdings with same index and columns as prices, but set all elements to 0
    trades = prices.copy()
    trades[:]=0
    trades['Cash'] = 0.00

    holdings = prices.copy()
    holdings[:]=0
    holdings['Cash'] = 0.00

    orderValue = 0.0            # Value of an order (Price of a share * number of shares bought/sold)
    shareValue = 0.0            # Price of a share
    numShares = 0               # Number of shares


    # Iterate through the orders dataframe, buy or sell shares depending on order type ('BUY' or 'SELL'), and
    # change share holdings and amount of cash available
    for i, row in orders.iterrows():

        numShares = row['Shares']
        stock = row['Symbol']
        orderType = row['Order']

        # Get value per share based on symbol and calculate value of total order
        shareValue = prices.ix[i, stock]
        orderValue = numShares * shareValue

        # Buy or sell depending on orderType
        if orderType == 'BUY':
            trades.loc[i, stock] += numShares
            trades.loc[i, 'Cash'] -= orderValue
            trades.loc[i, 'Cash'] -= commission
        else:
            trades.loc[i, stock] -= numShares
            trades.loc[i, 'Cash'] += orderValue
            trades.loc[i, 'Cash'] -= commission


        orderImpact = impact * numShares * shareValue
        trades.loc[i, 'Cash'] -= orderImpact
    """----------------------------------------------------------------------------------------"""

    # Update the holdings dataframe based on the trades made in orders
    holdings.ix[0,:] = trades.ix[0,:]
    holdings['Cash'][0] += start_val

    # Create dataframe for values with same index and columns and prices, set all elements to 0
    values = prices.copy()
    values[:] = 0
    values['Cash'] = 1.00

    # Update the holdings dataframe based on each day's trade
    for i in range(1, prices.shape[0]):
        holdings.ix[i,:] = holdings.ix[i-1,:] + trades.ix[i,:]

    # Calculate value of shares for each day - prices * holdings
    values = prices * holdings

    # Summarize to get total value of portfolio
    portvals = values.sum(axis=1)

    # Get stats for portfolio
    finalPortfolioValue = portvals[-1]
    cumReturn = (portvals[-1] / portvals[0]) - 1
    dailyRets = (portvals / portvals.shift(1)) - 1
    avgDailyRet = dailyRets.mean()
    stdDailyRet = dailyRets.std()

    print "Result for Portfolio containing: " + str(symbolsList)
    print ""
    print "Cumulative Return = " + str(cumReturn)
    print ""
    print "Average Daily Return = " + str(avgDailyRet)
    print ""
    print "Standard Deviation = " + str(stdDailyRet)
    print ""
    print "Final Portfolio Value = " + str(finalPortfolioValue)
    print "----------------------------------------------------"

    return portvals


  		   	  			    		  		  		    	 		 		   		 		  
def test_code():
    # this is a helper function you can use to test your code  		   	  			    		  		  		    	 		 		   		 		  
    # note that during autograding his function will not be called.  		   	  			    		  		  		    	 		 		   		 		  
    # Define input parameters  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
    # of = "./orders/orders2.csv"
    # sv = 1000000
  	#
    # # Process orders
    # portvals = compute_portvals(orders_file = of, start_val = sv)
    # if isinstance(portvals, pd.DataFrame):
    #     portvals = portvals[portvals.columns[0]] # just get the first column
    # else:
    #     "warning, code did not return a DataFrame"
  	#
    # # Get portfolio stats
    # # Here we just fake the data. you should use your code from previous assignments.
    # start_date = dt.datetime(2008,1,1)
    # end_date = dt.datetime(2008,6,1)
    # cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
    # cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    dateRange = pd.date_range(sd, ed)
    symbol = 'SPY'
    df_prices = get_data([symbol], dateRange)
    pricesSPY = df_prices['SPY']
    pricesSPY = pricesSPY / pricesSPY[0]     # Normalize to 1.0

    results = compute_portvals(pricesSPY, 100000, 0, 0)

    # start_date = sd
    # end_date = ed
    #
  	#
    # # Compare portfolio against $SPX
    # print "Date Range: {} to {}".format(start_date, end_date)
    # print
    # print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    # print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    # print
    # print "Cumulative Return of Fund: {}".format(cum_ret)
    # print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    # print
    # print "Standard Deviation of Fund: {}".format(std_daily_ret)
    # print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    # print
    # print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    # print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    # print
    # print "Final Portfolio Value: {}".format(portvals[-1])
  		   	  			    		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		   	  			    		  		  		    	 		 		   		 		  
    test_code()

