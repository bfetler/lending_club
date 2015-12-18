# autocorrelation time series (ACF, PACF, ~ARIMA)

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import collections
import os

plotdir = 'multivar_plots/'
if not os.access(plotdir, os.F_OK):
    os.mkdir(plotdir)

print 'reading csv data ...',
df = pd.read_csv('data/LoanStats3c.csv', header=1, low_memory=False)
print 'copying data ...',
# header=1 first line contains description
df2 = df.copy()
print 'done'

print 'adding log_income, issue_d_format ...',
# df2.dropna(inplace=True)  # drops 4 rows
df2['log_income'] = np.log10(df2.annual_inc)

# add time series objects
# converts string to datetime object in pandas:
df2['issue_d_format'] = pd.to_datetime(df2['issue_d']) 
print 'done'
print 'setting index ...',
dfts = df2.set_index('issue_d_format')   # class DataFrame
print 'done'

print 'groupby lambda ...',
year_month_summary = dfts.groupby(lambda x : x.year * 100 + x.month).count()
print 'done'
# groupby(lambda) applies lambda to index only
loan_count_summary = year_month_summary['issue_d']

# df2.dropna(inplace=True)  # drop 1 row w/ ANY, can't do stats on it
print 'df2 columns:', df2.columns
print 'df2 shape:', df2.shape
print 'df2 init\n', df2[:5]
d2list = ['issue_d','issue_d_format']
print 'df2 init issue_d:'
print df2[d2list][:5]
print df2[d2list][-5:]   # last 4 NaT's
# print 'home_ownership values:', set(df2['home_ownership'].tolist())

print 'dfts TIME SERIES'
print dfts[:5]
print dfts[-5:]    # last 4 NaT's
print 'dfts shape', dfts.shape, 'class', dfts.__class__    # DataFrame
print 'year_month_summary:', year_month_summary.__class__  # DataFrame
print year_month_summary
print 'loan_count_summary:', loan_count_summary.__class__  # Series
print loan_count_summary.shape
print loan_count_summary

# might this be faster, get count() only on one column, not all 57?
year2_month_summary = dfts['issue_d'].groupby(lambda x : x.year * 100 + x.month).count()
print 'y2s:', year2_month_summary.__class__, year2_month_summary.shape
print year2_month_summary
# drop missing dates - doesn't matter, NaN skipped by count() method
year2_month_summary = dfts['issue_d'].dropna()  # drops 4 rows
print 'y2s:', year2_month_summary.__class__, year2_month_summary.shape
year2_month_summary = year2_month_summary.groupby(lambda x : x.year * 100 + x.month).count()
print 'y2s:', year2_month_summary.__class__, year2_month_summary.shape
print year2_month_summary

print 'starting plot'
plt.clf()
# loan_count_summary.plot()     # 13 values, 1st is -101
year2_month_summary.plot()  # 12 values, w/o -101
plt.show()



