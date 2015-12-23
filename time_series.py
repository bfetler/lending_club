# autocorrelation time series (ACF, PACF, ARIMA)

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import collections
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
# df2['log_income'] = np.log10(df2.annual_inc)

# add time series objects
# converts string to datetime object in pandas:
# df2['issue_d_format'] = pd.to_datetime(df2['issue_d']) 
df2['issue_d_format'] = pd.to_datetime(df2['issue_d'], format='%b-%Y')
print 'done'
print 'setting index ...',
dfts = df2.set_index('issue_d_format')   # class DataFrame
print 'done'

print 'groupby orig lambda ...',
# year_month_summary = dfts.groupby(lambda x : x.year * 100 + x.month).count()
# year_month_summary = dfts.groupby(lambda x : x.year * 12 + x.month).count()
# groupby(lambda) applies lambda to index only
# loan_count_summary = year_month_summary['issue_d']
print 'done'

print 'groupby small_df lambda ...',
# might this be faster, get count() only on one column, not all 57?
# drop missing dates - doesn't matter, NaN skipped by count() method
loan_count2_summary = dfts['issue_d'].dropna()  # drops 4 rows
# loan_count2_summary = loan_count2_summary.groupby(lambda x : x.year * 100 + x.month).count()
# loan_count2_summary = loan_count2_summary.groupby(lambda x : x.year * 12 + x.month).count()
# loan_count2_summary = loan_count2_summary.groupby(lambda x : x).count()
# loan_count2_summary = loan_count2_summary.groupby().count() # fails
loan_count2_summary = loan_count2_summary.groupby(loan_count2_summary.index).count()
# only reason I can't use df.groupby('key') is it's a Series not DataFrame
# loan_count2_summary = loan_count2_summary + 0.0  # convert to float
loan_count2_summary = loan_count2_summary.map(lambda x : float(x))  # convert to float
print 'done'
# ok, seriously, I should change the x-axis variable to DateTime, not a stupid big int

# df2.dropna(inplace=True)  # drop 1 row w/ ANY, can't do stats on it
print 'df2 columns:', df2.columns
print 'df2 shape:', df2.shape
print 'df2 init\n', df2[:5]
d2list = ['issue_d','issue_d_format']
print 'df2 init issue_d:'
print df2[d2list][:10]
print df2[d2list][-10:]   # last 4 NaT's
print 'issue_d values:', set(df2['issue_d'].tolist())
print 'issue_d_format values:', set(df2['issue_d_format'].tolist())

print 'dfts TIME SERIES'
print dfts[:5]
print dfts[-5:]    # last 4 NaT's
print 'dfts shape', dfts.shape, 'class', dfts.__class__    # DataFrame
# print 'year_month_summary:', year_month_summary.__class__  # DataFrame
# print year_month_summary
# print 'loan_count_summary:', loan_count_summary.__class__  # Series
# print loan_count_summary.shape
# print loan_count_summary

print 'loan_count2_summary:', loan_count2_summary.__class__, loan_count2_summary.shape
print loan_count2_summary.index
print loan_count2_summary['2012-01-01'].__class__  # it's an int of course
print loan_count2_summary['2012-01-01']  # ok this works
print loan_count2_summary

print 'starting initial plots ...',
plt.clf()
# loan_count_summary.plot()     # 13 or 25 values, 1st is -101
loan_count2_summary.plot()  # 12 or 24 values, w/o -101
# plt.show()
plt.savefig(plotdir+'loan_monthly_count.png')

plt.clf()
sm.graphics.tsa.plot_acf(loan_count2_summary)
# plt.show()
plt.savefig(plotdir+'loan_monthly_acf_1.png')

plt.clf()
# sm.graphics.tsa.plot_pacf(loan_count2_summary)  # default fails
# sm.graphics.tsa.plot_pacf(loan_count2_summary, method='ywm')  # default fails
#   pacf.append(yule_walker(x, k, method=method)[0][-1])
#   File "/Users/bfetler/anaconda/lib/python2.7/site-packages/statsmodels/regression/linear_model.py", line 690, in yule_walker
#     X -= X.mean()                  # automatically demean's X
#   TypeError: Cannot cast ufunc subtract output from dtype('float64') to dtype('int64') with casting rule 'same_kind'
# sm.graphics.tsa.plot_pacf(loan_count2_summary, method='yw')  # fails same

# sm.graphics.tsa.plot_pacf(loan_count2_summary, method='ols')  # fails
#   File "/Users/bfetler/anaconda/lib/python2.7/site-packages/statsmodels/tsa/stattools.py", line 547, in pacf
#     ret = pacf_ols(x, nlags=nlags)
#   File "/Users/bfetler/anaconda/lib/python2.7/site-packages/statsmodels/tsa/stattools.py", line 502, in pacf_ols
#     res = OLS(x0[k:], xlags[k:,:k+1]).fit()
#   File "/Users/bfetler/anaconda/lib/python2.7/site-packages/statsmodels/base/data.py", line 246, in _check_integrity
#     if len(self.exog) != len(self.endog):
#   TypeError: len() of unsized objec

sm.graphics.tsa.plot_pacf(loan_count2_summary, method='ld')  # it works!
# flat cutoff, points above: +0 +1 -14 -15 +16 17 +18 +19, +22, -23
# sm.graphics.tsa.plot_pacf(loan_count2_summary, method='ldb')  # it works!
# flat cutoff, 1st two points above +0 +1

# plt.show()
plt.savefig(plotdir+'loan_monthly_pacf_ld.png')

plt.clf()
sm.graphics.tsa.plot_pacf(loan_count2_summary, method='ldb')  # it works!
# flat cutoff, 1st two points above +0 +1
plt.savefig(plotdir+'loan_monthly_pacf_ldb.png')

print 'done'

# for hints see http://statsmodels.sourceforge.net/devel/examples/notebooks/generated/tsa_arma.html

arima_mod100 = sm.tsa.ARIMA(loan_count2_summary, (1,0,0)).fit()
# print 'arima mod100 params:\n', arima_mod100.params
# print 'arimd mod100 summary:\n', arima_mod100.summary()
# print 'arimd mod100 predict:\n', arima_mod100.predict()
# print 'arimd mod100 resid:\n', arima_mod100.resid

# arima_mod100.resid.plot()
# plt.plot(arima_mod100.resid)
# qqplot(arima_mod100.resid, line='q')
# sm.graphics.tsa.plot_acf(arima_mod100.resid)
# sm.graphics.tsa.plot_pacf(arima_mod100.resid)
# r,q,p = sm.tsa.acf(arima_mod100.resid, qstat=True)
# print p

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

