from datetime import date, timedelta
import pandas_datareader.data as web
from prettytable import PrettyTable
from pandas.stats.api import ols
from sklearn import linear_model
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.stats import norm
from itertools import chain
import seaborn as sns
import pandas as pd
import numpy as np
import quandl
import math
import sys

class bcolors:
    HEADER    = '\033[95m'
    OKBLUE    = '\033[94m'
    OKGREEN   = '\033[92m'
    WARNING   = '\033[93m'
    FAIL      = '\033[91m'
    ENDC      = '\033[0m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    CYAN      = '\033[36m'

def getPortfolio():
	print( bcolors.FAIL + '\n**' + bcolors.CYAN + 
		'PYMD Efficient Frontier Portfolio Optimization' 
		+ bcolors.FAIL + '**' + bcolors.ENDC)
	stocks = [str(x) for x in input('Enter Desired Stocks: ').split()]
	[x.upper() for x in stocks]
	bench = input('Enter Desired Benchmark Index: ')
	bench.upper()
	interval = input('Enter Desired Price Interval (\'d\',\'w\', or \'m\'): ')
	start = input('Enter Start Date: ')
	return(stocks,bench,interval,start)

class OptimalFrontier:
	def __init__(self,stocks,index,interval,start):
		self.stocks = [x.upper() for x in stocks]
		self.index = index.upper()
		self.interval = interval
		self.start = start
		if self.interval == 'm':
			self.multiplier = 12
		elif self.interval == 'w':
			self.multiplier = 52
		else:
			self.multiplier = 252
		self.fetchData()

	def fetchData(self):
		#download daily price data for each of the stocks in the portfolio
		try:
			self.sec_data = web.get_data_yahoo(self.stocks,start= self.start,interval=self.interval)['Adj Close']
			self.index_data = web.get_data_yahoo(self.index,start= self.start,interval=self.interval)['Adj Close']
		except:
			print(bcolors.FAIL + "Error: Ticker Not Found!" + bcolors.ENDC);sys.exit()

		print(bcolors.HEADER + '\nFetching Fama French Factor Data...' + bcolors.ENDC)
		self.ff = (quandl.get("KFRENCH/FACTORS_D", authtoken="YmxsK5nrXQPoE2-nUWxp")['RF']/100)

		print(bcolors.HEADER + '\nFetching 10 Year Treasury Rates...' + bcolors.ENDC)
		self.trate = (quandl.get("USTREASURY/YIELD", authtoken="YmxsK5nrXQPoE2-nUWxp",
			start_date=date.today()-timedelta(5))['10 YR']/100)
 
	def getCompoundedReturns(self):
		self.sec_returns = np.log(self.sec_data).diff()
		self.inx_returns = np.log(self.index_data).diff()

	def computeAnalytics(self):
		print(bcolors.WARNING + '\nComputing Analytics...' + bcolors.ENDC)
		#calculate mean daily return and covariance of daily returns
		self.mu_ret     = self.sec_returns.mean()
		self.med_ret    = self.sec_returns.median()
		self.std_ret    = self.sec_returns.std()
		self.skew_ret   = self.sec_returns.skew()
		self.kurt_ret   = self.sec_returns.kurtosis()
		self.cov_matrix = self.sec_returns.cov()
		self.cor_matrix = self.sec_returns.corr()
		self.ann_ret    = self.mu_ret*self.multiplier
		self.ann_std    = self.std_ret*math.sqrt(self.multiplier)

	def prepCAPMRegression(self):
		#prepare regressions for CAPM alpha & beta
		self.rsec_rf = self.sec_returns.subtract(self.ff,axis=0)
		self.rinx_rf = self.inx_returns.subtract(self.ff,axis=0)
		self.rsec_rf = self.rsec_rf.dropna()
		self.rinx_rf = self.rinx_rf.dropna()
		
	def computeCAPM(self):
		print(bcolors.WARNING + '\nComputing CAPM Regressions...' + bcolors.ENDC)
		#regressions to derive beta and alphas
		regr = linear_model.LinearRegression()
		self.r2 = []
		self.alpha = []
		self.beta = []
		for col in self.rsec_rf:
			regr.fit(self.rinx_rf.values.reshape((len(self.rinx_rf),1)),self.rsec_rf[col])
			self.r2.append(regr.score(self.rinx_rf.values.reshape((len(self.rinx_rf),1)),self.rsec_rf[col]))
			self.alpha.append(regr.intercept_)
			self.beta.append(regr.coef_)

	def computeRatios(self):
		#compute a few more portfolio stats
		self.at_risk = norm.ppf(0.05,loc=self.ann_ret,scale=self.ann_std)
		self.sec_sharpe = (self.ann_ret-self.trate[-1])/self.ann_std
		#compute sortino ratio
		target = .03/252
		rt = self.mu_ret-target
		def minimum(a,b):
			if a < b:
				return a
			else:
				return b
		vfunc = np.vectorize(minimum)
		tdd = np.sqrt(pd.DataFrame(vfunc(0,(self.sec_returns-target))**2).mean())
		tdd.index = rt.index
		self.sortino = rt.divide(tdd,axis=0)*math.sqrt(self.multiplier)

	def computeEqualPortfolio(self):
		# Compute Equally Weighted Portfolio Returns
		self.start_weights = np.asarray([1/len(self.stocks)] * len(self.stocks))
		self.portfolio_return = np.dot(self.start_weights,self.mu_ret)*self.multiplier
		self.portfolio_std = np.sqrt(np.dot(self.start_weights.T,np.dot(self.cov_matrix,self.start_weights)))*np.sqrt(self.multiplier)
		self.portfolio_at_risk = np.dot(self.start_weights,self.at_risk)
		self.portfolio_sharpe = (self.portfolio_return-self.trate[-1])/self.portfolio_std

	def buildStrings(self):
		self.avg_returns     = ['Avg. Returns']       + list('%.3f%%' % elem for elem in (self.mu_ret*100))
		self.med_returns     = ['Med. Returns']       + list('%.3f%%' % elem for elem in (self.med_ret*100))
		self.std_returns     = ['Std. Deviation']     + list('%.3f%%' % elem for elem in (self.std_ret*100))
		self.skewness        = ['Skew']               + list('%.2f' % elem for elem in (self.skew_ret))
		self.kurt            = ['Kurtosis']           + list('%.2f' % elem for elem in (self.kurt_ret))
		self.annual_returns  = ['Annual Returns']     + list('%.2f%%' % elem for elem in (self.ann_ret*100))
		self.annual_std      = ['Annual Std Dev']     + list('%.2f%%' % elem for elem in (self.ann_std*100))

		self.r2_tab          = ['R2 vs '+ self.index] + list('%.2f' % elem for elem in self.r2)
		self.alpha_tab       = ['Alpha']              + list('%.2f' % elem for elem in self.alpha)
		self.beta_tab        = ['Adj Beta']           + list('%.2f' % elem for elem in self.beta)
		self.value_risk      = ['VaR 5%']             + list('%.2f%%' % elem for elem in (self.at_risk*100))
		self.sharp_tab       = ['Sharpe Ratio']       + list('%.3f' % elem for elem in (self.sec_sharpe))
		self.sortino_tab     = ['Sortino Ratio']      + list('%.3f' % elem for elem in (self.sortino))

		self.ret_port        = ['Annual Return']      + ['%.2f%%' % (self.portfolio_return*100)]
		self.std_port        = ['Annual Std Dev']     + ['%.2f%%' % (self.portfolio_std*100)]
		self.value_risk_port = ['VaR 5%']             + ['%.2f%%' % (self.portfolio_at_risk*100)]
		self.sharpe_port     = ['Sharpe Ratio']       + ['%.3f' % (self.portfolio_sharpe)]

	def buildTables(self):
		headers = [''] + list(self.sec_data.columns)
		self.tab = PrettyTable(headers)
		self.tab.align[''] = "l"
		self.tab.add_row(self.avg_returns)
		self.tab.add_row(self.med_returns)
		self.tab.add_row(self.std_returns)
		self.tab.add_row(self.skewness)
		self.tab.add_row(self.kurt)
		self.tab.add_row(self.annual_returns)
		self.tab.add_row(self.annual_std)

		#table number two
		headers2 = [''] + list(self.sec_data.columns)
		self.tab2 = PrettyTable(headers)
		self.tab2.align[''] = "l"
		self.tab2.add_row(self.r2_tab)
		self.tab2.add_row(self.alpha_tab)
		self.tab2.add_row(self.beta_tab)
		self.tab2.add_row(self.value_risk)
		self.tab2.add_row(self.sharp_tab)
		self.tab2.add_row(self.sortino_tab)

		w = ((str('%.2f%%' % (1/len(self.stocks)*100)+' / '))*len(self.stocks))[:-2]
		headers3 = ['Portfolio #1'] + [w]
		self.tab3 = PrettyTable(headers3)
		self.tab3.align['Portfolio #1'] = "l"
		self.tab3.add_row(self.ret_port)
		self.tab3.add_row(self.std_port)
		self.tab3.add_row(self.value_risk_port)
		self.tab3.add_row(self.sharpe_port)

	def displayTables(self):
		print(bcolors.UNDERLINE + "\nTable 1: Analysis of Securities" + bcolors.ENDC)
		print(self.tab)
		print(bcolors.UNDERLINE + "\nTable 2: Portfolio Indicators" + bcolors.ENDC)
		print(self.tab2)
		print(bcolors.UNDERLINE + "\nTable 3: Equally Weighted Portfolio" + bcolors.ENDC)
		print(self.tab3)

	def monteCarlo(self):
		print( bcolors.FAIL + '\n**' + bcolors.CYAN + 
		'Monte Carlo Portfolio Optimization' + bcolors.FAIL + '**' + bcolors.ENDC)
		num_iterations = int(input('Enter Number of Simulations: '))
		self.results = np.zeros((4+len(self.stocks)-1,num_iterations))
		for i in range(num_iterations):
			weights = np.array(np.random.random(len(self.stocks)))
			weights /= np.sum(weights)
			sim_return = np.sum(self.mu_ret * weights) * self.multiplier
			sim_std = np.sqrt(np.dot(weights.T,np.dot(self.cov_matrix, weights))) * np.sqrt(self.multiplier)
			self.results[0,i] = sim_return
			self.results[1,i] = sim_std
			self.results[2,i] = (self.results[0,i]-self.trate[-1])/self.results[1,i]
			for j in range(len(weights)):
				self.results[j+3,i] = weights[j]

	def cleanMonteCarloData(self):
		#convert results array to Pandas DataFrame
		self.results_frame = pd.DataFrame(self.results.T,columns=['ret','stdev','sharpe']+list(self.sec_returns.columns))
		self.results_frame['ret']   = self.results_frame['ret'] * 100
		self.results_frame['stdev'] = self.results_frame['stdev'] * 100
		self.results_frame[self.stocks]  = self.results_frame[self.stocks] * 100
		self.max_sharpe_port        = self.results_frame.iloc[self.results_frame['sharpe'].idxmax()]
		self.min_vol_port           = self.results_frame.iloc[self.results_frame['stdev'].idxmin()]

	def plotFrontier(self):
		plt.scatter(self.results_frame.stdev,self.results_frame.ret,c=self.results_frame.sharpe,cmap='RdYlBu')
		plt.title('Efficient Frontier Portfolio - ' + str(self.stocks))
		plt.xlabel('Volatility %')
		plt.ylabel('Returns %')
		plt.colorbar()
		plt.scatter(self.max_sharpe_port[1],self.max_sharpe_port[0],marker='*',color='mediumspringgreen',s=200)
		plt.scatter(self.min_vol_port[1],self.min_vol_port[0],marker='*',color='salmon',s=200)
		from io import BytesIO
		import base64, urllib
		my_stringIObytes = BytesIO()
		plt.savefig(my_stringIObytes, format='png',bbox_inches='tight')
		my_stringIObytes.seek(0)
		self.my_base64 = base64.b64encode(my_stringIObytes.read())
		#plt.savefig('static/efficientFrontier.png', format='png', dpi=1000, bbox_inches='tight')
		plt.show()

	def buildPortfolioStrings(self):
		self.max_sharpe_port.loc['ret']       = ('%.3f%%' % (self.max_sharpe_port['ret']))
		self.max_sharpe_port.loc['stdev']     = ('%.3f%%' % (self.max_sharpe_port['stdev']))
		self.max_sharpe_port.loc['sharpe']    = ('%.3f' % (self.max_sharpe_port['sharpe']))
		self.max_sharpe_port.loc[self.stocks] = ['%.2f%%' % elem for elem in (self.max_sharpe_port.loc[self.stocks])]

		self.min_vol_port.loc['ret']       = ('%.3f%%' % (self.min_vol_port['ret']))
		self.min_vol_port.loc['stdev']     = ('%.3f%%' % (self.min_vol_port['stdev']))
		self.min_vol_port.loc['sharpe']    = ('%.3f' % (self.min_vol_port['sharpe']))
		self.min_vol_port.loc[self.stocks] = ['%.2f%%' % elem for elem in (self.min_vol_port.loc[self.stocks])]

	def buildPortfolioTables(self):
		headers4 = ['Return','Std Dev','Sharpe']+list(self.sec_returns.columns)
		self.tab4 = PrettyTable(headers4)
		self.tab4.align['']="l"
		self.tab4.add_row(self.max_sharpe_port)

		headers5 = ['Return','Std Dev','Sharpe']+list(self.sec_returns.columns)
		self.tab5 = PrettyTable(headers5)
		self.tab5.align['']="l"
		self.tab5.add_row(self.min_vol_port)

	def displayPortfolioTables(self):
		print(bcolors.UNDERLINE + "\nTable 4: Maximum Sharpe Portfolio" + bcolors.ENDC)
		print(self.tab4)
		print(bcolors.UNDERLINE + "\nTable 5: Minimum Variance Portfolio" + bcolors.ENDC)
		print(self.tab5)

	def generateHTML(self):
		from jinja2 import Template, Environment, FileSystemLoader
		lines = [self.tab.get_html_string(format=True),self.tab2.get_html_string(format=True),
		self.tab3.get_html_string(format=True),self.tab4.get_html_string(format=True),self.tab5.get_html_string(format=True)]
		# Render html file
		env = Environment(loader=FileSystemLoader('templates'))
		template = env.get_template('template.html')
		output_from_parsed_template = template.render(table_1=lines[0],table_2=lines[1],table_3=lines[2],
			table_4=lines[3],table_5=lines[4],plot=self.my_base64.decode('utf8'),stocks=str(self.stocks),headingdate=date.today())
		# to save the results
		with open("OutputAnalysis.html", "w") as fh:
		    fh.write(output_from_parsed_template)

	def generatePDF(self):
		from weasyprint import HTML
		HTML('OutputAnalysis.html').write_pdf('OutputAnalysis.pdf')

	def generateEmail(self):
		import smtplib
		from email.mime.multipart import MIMEMultipart
		from email.mime.text import MIMEText
		from email.mime.base import MIMEBase
		from email import encoders
		
		emailChoice = input(bcolors.BOLD + '\nWould you like to recieve an automated email report? [Y/n]: ' + bcolors.ENDC)
		if(emailChoice=='N' or emailChoice=='n' or 
			emailChoice=='no' or emailChoice=='No' or emailChoice=='NO'):
			print(bcolors.OKBLUE + 'Goodbye' + bcolors.ENDC);sys.exit()

		else:
			emailAddr = input(bcolors.BOLD + bcolors.OKBLUE + 'Email: ' + bcolors.ENDC)
			fromaddr = "pymd.investments@gmail.com"
			toaddr = emailAddr
			 
			msg = MIMEMultipart()
			 
			msg['From'] = fromaddr
			msg['To'] = toaddr
			msg['Subject'] = "Automated Portfolio Allocation Report"
			import codecs
			f=codecs.open("OutputAnalysis.html", 'r')
			body = f.read()
			msg.attach(MIMEText(body, 'html'))
			 
			filename = "OutputAnalysis.html"
			attachment = open("OutputAnalysis.html", "rb")
			 
			part = MIMEBase('application', 'octet-stream')
			part.set_payload((attachment).read())
			encoders.encode_base64(part)
			part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
			 
			msg.attach(part)
			 
			server = smtplib.SMTP('smtp.gmail.com', 587)
			server.starttls()
			server.login(fromaddr, "Pymdinvestments1234")
			text = msg.as_string()
			server.sendmail(fromaddr, toaddr, text)
			server.quit()
			print(bcolors.WARNING + "Email Sent Successfully!" + bcolors.ENDC)

def main():
    s,b,i,st = getPortfolio()
    pt = OptimalFrontier(s,b,i,st)
    pt.getCompoundedReturns()
    pt.computeAnalytics()
    pt.prepCAPMRegression()
    pt.computeCAPM()
    pt.computeRatios()
    pt.computeEqualPortfolio()
    pt.buildStrings()
    pt.buildTables()
    pt.displayTables()
    pt.monteCarlo()
    pt.cleanMonteCarloData()
    pt.plotFrontier()
    pt.buildPortfolioStrings()
    pt.buildPortfolioTables()
    pt.displayPortfolioTables()
    pt.generateHTML()
    pt.generateEmail()

if __name__ == "__main__":
    main()



