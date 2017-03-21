import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from scipy.stats import norm, kurtosis
from pandas_datareader import data
from tabulate import tabulate

print("PYMD Monte Carlo Simulation")
stock = input('Please enter a ticker to evaluate: ')
date = input('Please enter a start date (e.g. 1/13/2000): ')
sec = data.DataReader(stock,'yahoo',start=date)
print('')

# compound annual growth (CAGR) which will give us \mu
days = (sec.index[-1] - sec.index[0]).days
cagr = ((((sec['Adj Close'][-1])/sec['Adj Close'][1])) ** (365.0/days)) - 1
print ('CAGR = ', str(round(cagr,4)*100)+"%")
mu = cagr

# caclulate annual volatility
sec['Returns'] = sec['Adj Close'].pct_change()
vol = sec['Returns'].std()*math.sqrt(252)
print ("Annual Volatility = ",str(round(vol,4)*100)+"%")

result = []

S = sec['Adj Close'][-1]
T = 252

# daily returns using random normal dist.

for i in range(10000):
    daily_returns = np.random.normal(mu/T,vol/math.sqrt(T),T)+1
    price_list = [S]

    for x in daily_returns:
        price_list.append(price_list[-1]*x)

    plt.plot(price_list)
    result.append(price_list[-1])

print('')
print("Current Price: $" + str(round(sec['Adj Close'][-1],2)))
print("Number of Iterations: 10000")

mean = np.mean(result)
stdev = np.std(result)
median = np.median(result)
skew = (3*(np.mean(result)-np.median(result)))/(np.std(result))
kurt = kurtosis(result,0,False,True)
ci_5 = np.percentile(result,5)
ci_95 = np.percentile(result,95)

print(tabulate([["Mean",'%.2f' % mean],
               ["Median",'%.2f' % median],
               ["StDev",'%.2f' % stdev],
               ["Skewness",'%.2f' % skew],
               ["Kurtosis",'%.2f' % kurt],
               ["5% CI",'%.2f' % ci_5],
               ["95% CI",'%.2f' % ci_95]],
      headers=['Model','Statistics'],tablefmt='orgtbl'))

# Line Chart
plt.xlim((0,252))
plt.title('Monte Carlo Simulation of ' + stock)
plt.ylabel('Simulated Price ($)')
plt.xlabel('Trading Days in One Year')
plt.grid(True)
plt.show()

# Histogram
plt.title('Histogram of Simulated Prices - ' + stock)
plt.ylabel('Number of Simulations')
plt.xlabel('Stock Price ($)')
plt.hist(result,bins=50,color='#3b5c91')
plt.grid(True)
plt.axvline(np.percentile(result,5), color='r', linestyle='dashed', linewidth=2)
plt.axvline(np.percentile(result,95), color='r', linestyle='dashed', linewidth=2)
plt.show()

