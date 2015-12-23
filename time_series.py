# autocorrelation time series (ACF, PACF, ARIMA)

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
import matplotlib.pyplot as plt
# import collections
import os

plotdir = 'time_series_plots/'
if not os.access(plotdir, os.F_OK):
    os.mkdir(plotdir)

print 'reading csv data ...',
df = pd.read_csv('data/LoanStats3b.csv', header=1, low_memory=False)
print 'copying data ...',
# header=1 first line contains description
df2 = df.copy()
print 'done'

print 'adding issue_d_format ...',
# df2.dropna(inplace=True)  # drops 4 rows

# add time series objects
# converts string to datetime object in pandas:
df2['issue_d_format'] = pd.to_datetime(df2['issue_d'], format='%b-%Y')
# weird - without format it uses current day-of-month, with format it's 01
dfts = df2.set_index('issue_d_format')   # class DataFrame
print 'done'
print 'dfts columns:', dfts.columns
# d2list = ['issue_d','issue_d_format','annual_inc','loan_amnt','int_rate','term']
d2list = ['issue_d','annual_inc','loan_amnt','int_rate','term']

print 'groupby index ...',
# keep only issue_d (loan issue date) rather than all 56 columns, may be faster
# drop missing dates - doesn't matter, NaN skipped by count() method
dfts = dfts['issue_d'].dropna()  # keep only 1 column, drop 4 rows, now Series
# dfts = dfts[d2list].dropna()  # keep only 1 column, drop 4 rows, DataFrame
# dfts class is now Series not DataFrame
# loan_count = loan_count.groupby(lambda x : x.year * 12 + x.month).count()
# loan_count = loan_count.groupby(lambda x : x).count()
loan_count = dfts.groupby(dfts.index).count()
# only reason I can't use df.groupby('key') is it's a Series not DataFrame

# loan_count = loan_count + 0.0  # convert to float - shortcut
loan_count = loan_count.apply(lambda x : float(x))  # convert to float
print 'done'

# df2.dropna(inplace=True)  # drop 1 row w/ ANY, can't do stats on it
# print 'df2 columns:', df2.columns
print 'df2 shape:', df2.shape, 'class', df2.__class__
# print df2.describe()
# print 'df2 init\n', df2[:5]
print 'df2 init issue_d:'
print df2[d2list][:8]
print df2[d2list][-8:]   # last 4 NaT's
print 'issue_d values:', set(df2['issue_d'].tolist())
print 'issue_d_format values:', set(df2['issue_d_format'].tolist())

# print 'dfts TIME SERIES'
# print dfts[:5]
# print dfts[-5:]    # last 4 NaT's
print 'dfts shape', dfts.shape, 'class', dfts.__class__    # Series

print 'loan_count:', loan_count.__class__, loan_count.shape  # Series
print loan_count.index
print loan_count['2012-01-01']  # ok this works
print loan_count

print 'starting initial plots ...',
plt.clf()
loan_count.plot()  # 12 or 24 values, w/o -101
# time axis labels need adjusting
# plt.show()
plt.savefig(plotdir+'loan_monthly_count.png')

plt.clf()
sm.graphics.tsa.plot_acf(loan_count)
# plt.show()
plt.savefig(plotdir+'loan_monthly_acf.png')

plt.clf()
sm.graphics.tsa.plot_pacf(loan_count)  # default fails w/ ints, ok w/ dates
# sm.graphics.tsa.plot_pacf(loan_count, method='ywm')  # default fails w/ ints
# sm.graphics.tsa.plot_pacf(loan_count, method='yw')   # fails w/ ints
# sm.graphics.tsa.plot_pacf(loan_count, method='ld')   # works
# sm.graphics.tsa.plot_pacf(loan_count, method='ldb')  # works
# sm.graphics.tsa.plot_pacf(loan_count, method='ols')  # fails w/ ints, dates

# plt.show()
plt.savefig(plotdir+'loan_monthly_pacf.png')

print 'done'

# for hints see http://statsmodels.sourceforge.net/devel/examples/notebooks/generated/tsa_arma.html

print 'arima model 100'

arima_mod100 = sm.tsa.ARIMA(loan_count, (1,0,0)).fit()
# print 'arima mod100 params:\n', arima_mod100.params
print 'arima mod100 summary:\n', arima_mod100.summary()
print 'arima mod100 predict:\n', arima_mod100.predict()
print 'arima mod100 resid:\n', arima_mod100.resid

plt.clf()
arima_mod100.resid.plot()
plt.savefig(plotdir+'arima_mod100_resid.png')

plt.clf()
plt.plot(arima_mod100.resid)
plt.savefig(plotdir+'arima_mod100_resid2.png')

plt.clf()
qqplot(arima_mod100.resid, line='q')
plt.savefig(plotdir+'arima_mod100_qqplot.png')

plt.clf()
sm.graphics.tsa.plot_acf(arima_mod100.resid)
plt.savefig(plotdir+'arima_mod100_acf.png')

plt.clf()
sm.graphics.tsa.plot_pacf(arima_mod100.resid)
plt.savefig(plotdir+'arima_mod100_pacf.png')

r,q,p = sm.tsa.acf(arima_mod100.resid, qstat=True)
print 'arima mod100 p-values', p

'''
data from 2012-01 to 2013-12, dataset 3b, more or less linear
24145     2602
24146     2560
24147     2914
24148     3230
24149     3400
24150     3817
24151     4627
24152     5419
24153     6087
24154     6263
24155     6382
24156     6066
24157     6872
24158     7561
24159     8273
24160     9419
24161    10350
24162    10899
24163    11910
24164    12674
24165    12987
24166    14115
24167    14676
24168    15020
data from 2014-01 to 2014-12, dataset 3c, more or less random
24169    15628
24170    15269
24171    16513
24172    19071
24173    19099
24174    17179
24175    29306
24176    18814
24177    10606
24178    38783
24179    25054
24180    10307
'''

