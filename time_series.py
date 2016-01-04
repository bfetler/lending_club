# autocorrelation time series (ACF, PACF, ARIMA)

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
import matplotlib.pyplot as plt
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
# dfts class is now Series not DataFrame
# loan_count = loan_count.groupby(lambda x : x.year * 12 + x.month).count()
loan_count = dfts.groupby(dfts.index).count()
# can't use df.groupby('key'), it's a Series not DataFrame

# loan_count = loan_count + 0.0  # convert to float - shortcut
loan_count = loan_count.apply(lambda x : float(x))  # convert to float
print 'done'

# print 'df2 columns:', df2.columns
print 'df2 shape:', df2.shape, 'class', df2.__class__
# print df2.describe()
# print 'df2 init\n', df2[:5]
print 'df2 init issue_d:'
print df2[d2list][:8]
print df2[d2list][-8:]   # last 4 NaT's
print 'issue_d values:', set(df2['issue_d'].tolist())
print 'issue_d_format values:', set(df2['issue_d_format'].tolist())

print 'dfts shape', dfts.shape, 'class', dfts.__class__    # Series

print 'loan_count:', loan_count.__class__, loan_count.shape  # Series
print loan_count.index
print loan_count['2012-01-01']  # print 1st float
print loan_count

print 'starting initial plots ...',
newsize=(8, 5.2)  # crop out time axis variable
# newsize=(8, 7)  # include full variables, default is (8, 6)
plt.clf()
# loan_count.plot()  # 12 or 24 values w/o NaN
# loan_count.plot(figsize=(8, 7))  # 24 values w/o NaN; increase y figsize
loan_count.plot(figsize=newsize)  # 24 values w/o NaN; default figsize(8,6)
# decrease y-axis figsize to crop out time axis variable
plt.title('Monthly Loan Count by Issue Date, 2012-2013')
plt.savefig(plotdir+'loan_monthly_count.png')

plt.clf()
sm.graphics.tsa.plot_acf(loan_count)
plt.text(12 ,0.5, 'Autocorrelation of monthly\n   loan count, 2012-2013')
# plt.show()
plt.savefig(plotdir+'loan_monthly_acf.png')

plt.clf()
sm.graphics.tsa.plot_pacf(loan_count)  # default fails w/ ints, ok w/ dates
# sm.graphics.tsa.plot_pacf(loan_count, method='ywm')  # default fails w/ ints
# sm.graphics.tsa.plot_pacf(loan_count, method='yw')   # fails w/ ints
# sm.graphics.tsa.plot_pacf(loan_count, method='ld')   # works
# sm.graphics.tsa.plot_pacf(loan_count, method='ldb')  # works
# sm.graphics.tsa.plot_pacf(loan_count, method='ols')  # fails w/ ints, dates

plt.text(8 ,0.7, '     Partial autocorrelation of\nmonthly loan count, 2012-2013')
plt.savefig(plotdir+'loan_monthly_pacf.png')

print 'done'

# for hints see http://statsmodels.sourceforge.net/devel/examples/notebooks/generated/tsa_arma.html


def arima_analysis(p, d, q):
    '''p is the number of autoregressive terms
    d is the number of nonseasonal differences
    q is the number of lagged forecast errors'''

    if p<0 or d<0 or q<0:
        print 'arima_analysis argument cannot be less than zero'
        return

    label  = str(p) + str(d) + str(q)
    plabel = '(' + str(p) + ',' + str(d) + ',' + str(q) + ')'
    alabel = '\narima model ' + label
    print alabel + ' analysis'
    arima_mod = sm.tsa.ARIMA(loan_count, (p, d, q)).fit()
#   print alabel+' params:\n',  arima_mod.params
    print alabel+' summary:\n', arima_mod.summary()

    print alabel+' predict:\n', arima_mod.predict()
    if d>0:  # fails if d>1, need multiple cumsum?
        pred = arima_mod.predict().cumsum() + loan_count['2012-01-01']
        print alabel+' predict fix:\n', pred

    print alabel+' resid:\n',   arima_mod.resid

    rr,rq,rp = sm.tsa.acf(arima_mod.resid, qstat=True)
    print alabel + ' p-values:\n', rp

    plotpath = plotdir + 'arima_mod' + label + '_'
    plt.clf()
    arima_mod.resid.plot(figsize=newsize)
#   plt.text(arima_mod.resid.index[13], -600, 'Residual of ARIMA'+plabel+' fit\nof monthly loans 2012-2013')
    coord = -3800 * p + 3200 * d
    plt.text(arima_mod.resid.index[13], coord, 'Residual of ARIMA'+plabel+' fit\nof monthly loans 2012-2013')
    plt.savefig(plotpath + 'resid.png')
#   plt.clf()
#   plt.plot(arima_mod.predict().index, arima_mod.resid)  # ok, needs work
#   plt.savefig(plotpath + 'resid2.png')
    plt.clf()
    qqplot(arima_mod.resid, line='q', fit=True)
    plt.title('Quantile-Quantile Plot')
    plt.text(0.0, -1.5, 'QQ plot of ARIMA'+plabel+' fit\nof monthly loans 2012-2013')
    plt.savefig(plotpath + 'qqplot.png')
    plt.clf()
    sm.graphics.tsa.plot_acf(arima_mod.resid)
    plt.text(6 ,0.7, 'Autocorrelation of ARIMA'+plabel+' fit\nresidual, monthly loans 2012-2013')
    plt.savefig(plotpath + 'acf.png')
    plt.clf()
    sm.graphics.tsa.plot_pacf(arima_mod.resid)
    plt.text(5 ,0.7, 'Partial autocorrelation of ARIMA'+plabel+'\nfit residual, monthly loans 2012-2013')
    plt.savefig(plotpath + 'pacf.png')

    plt.clf()
    if d>0:    # fails if d>1, needs multiple cumsum?
        pred = arima_mod.predict().cumsum() + loan_count['2012-01-01']
        pred.plot(figsize=newsize, style='r-') 
    else:
        arima_mod.predict().plot(figsize=newsize, style='r-') 
    loan_count.plot(figsize=newsize)
    plt.title('Monthly Loan Count 2012-2013')
    plt.legend(['ARIMA'+plabel+' fit prediction','Actual number of loans'], loc='upper left')
    plt.savefig(plotpath + 'predict.png')
#   print 'loan len, predict len', len(loan_count.values), len(arima_mod100.predict())
#   print loan_count.index   # Freq None
#   print arima_mod.predict().index  # Freq MS

    del arima_mod  # clean from memory, seems to help w/ multiple fits
    os.remove('iterate.dat')


print 'Starting ARIMA analyses'
arima_analysis(1, 0, 0)
arima_analysis(1, 1, 0)
arima_analysis(2, 0, 0)
arima_analysis(2, 1, 0)
# help(arima_analysis)   # test help output
print 'ARIMA analyses done.'
print '\nCONCLUSION: It seems that the ARIMA(1, 1, 0) model best fits the data.'
print 'Other models tried seem to overfit, leading to no significant improvement in p-values and autocorrelations.'
print 'ARIMA(1, 0, 0) has better p-values but fits with all nan for stderr.  Since the data clearly has a linear trend and isn\'t stationary, d=0 is not a good match.'



'''
Note the data is somewhat artificial.  Data from 2012-01 to 2013-12
is more or less linear, while data from 2014-01 to 2014-12 seems random.

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

