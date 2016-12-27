import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from scipy.stats import norm, kurtosis
from pandas_datareader import data

print("DJM Monte Carlo Simulation")
stock = input('Please enter a ticker to evaluate: ')
date = input('Please enter a start date (e.g. 1/13/2000): ')
try:
    sec = data.DataReader(stock,'yahoo',start=date)
except IOError:
    print('There was an error')

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
print("Number of Iterations: 10000")
print("Mean:",round(np.mean(result),2))
print("Std. Dev:",round(np.std(result),2))
print("Median:", round(np.median(result),2))
print("Skewness:",round((3*(np.mean(result)-np.median(result)))/(np.std(result)),2))
print("Kurtosis:",round(kurtosis(result),2))
print("5% CI:", round(np.percentile(result,5),2))
print("95% CI:",round(np.percentile(result,95),2))

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



