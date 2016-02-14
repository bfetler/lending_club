# cross validation of linear regression

import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
import re
import os

# loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
loansData = pd.read_csv('data/loansData.csv')
loansData.dropna(inplace=True)

# plotdir = 'linear_cross_plots/'
# if not os.access(plotdir, os.F_OK):
#     os.mkdir(plotdir)

pat = re.compile('(.*)-(.*)')  # ()'s return two matching fields

def splitSum(s):
#   t = s.split('-')
    t = re.findall(pat, s)[0]
    return (int(t[0]) + int(t[1])) / 2

loansData['Interest.Rate'] = loansData['Interest.Rate'].apply(lambda s: float(s.rstrip('%')))
loansData['Loan.Length'] = loansData['Loan.Length'].apply(lambda s: int(s.rstrip(' months')))
loansData['Debt.To.Income.Ratio'] = loansData['Debt.To.Income.Ratio'].apply(lambda s: float(s.rstrip('%')))
loansData['FICO.Average'] = loansData['FICO.Range'].apply(splitSum)
# loansData['FICO.Average'] = loansData['FICO.Range'].map(splitSum)

print 'loansData head\n', loansData[:5]
print '\nloansData basic stats\n', loansData.describe()   # print basic stats

def mean_abs_error(resid):
    return np.mean(np.abs(resid))

def median_abs_error(resid):
    return np.median(np.abs(resid))

def metric_summary(fit):
    mae, mad = mean_abs_error(fit.resid), median_abs_error(fit.resid)
    rsq, mse = fit.rsquared, fit.mse_resid
    rsqa, mset, ssr = fit.rsquared_adj, fit.mse_total, fit.ssr
#   print 'fit mae %g, mad %g, mse %g, rsq %g' % (mae, mad, mse, rsq)
#   print 'fit rsq_adj %g, mse_total %g, ssr %g' % (rsqa, mset, ssr)
    return [mae, mad, mse, rsq]

def print_metric_stats(label, metrics):
    mn = np.mean(metrics)
    sd = np.std(metrics)
    print '%s mean %g, sd %g, pct=%g' % (label, mn, sd, sd/mn)

# calculate linear regression example
# equation: InterestRate = b + a1 * FICO.Average + a2 * Loan.Amount
# full data
y  = np.matrix(loansData['Interest.Rate']).transpose()
x1 = np.matrix(loansData['Amount.Requested']).T
x2 = np.matrix(loansData['FICO.Average']).T
print 'IntRate matrix', y[:5]
print 'Amt matrix', x1[:5]
print 'FICO matrix', x2[:5]
X = sm.add_constant( np.column_stack([x1, x2]) )  # x's plus constant
model = sm.OLS(y, X)   # ordinary least squares
f = model.fit()
print 'full model fit summary\n', f.summary()
met = metric_summary(f)
print 'metrics', met
# print 'full fit r_squared %g, rsq_adj %g, mse_resid %g, mse_total %g, ssr %g' % \
#        (f.rsquared, f.rsquared_adj, f.mse_resid, f.mse_total, f.ssr)
# print 'full fit mean_abs %g, median_abs %g' % (mean_abs_error(f.resid), median_abs_error(f.resid))

# plot exploratory data
# plt.clf()
# loansData.groupby('Loan.Purpose').hist()     # more useful
# plt.savefig(plotdir+'LoanPurpose_Histogram.png')

# test KFold fitting
# split data into ten folds, repeat OLS, compute metrics on each one
print('\nKFold 10, len X %d, len y %d' % (len(X), len(y)) )

# use np.array instead of list for metrics, if large data
metrics = []

kf = KFold(len(X), n_folds=10)
for train, test in kf:
#   print("test (len %d) %s, train (len %d) %s %s %s %s %s" % (len(test), test[:3], \
#          len(train), train[0:3], train[500:503], train[1000:1003], \
#          train[1500:1503], train[2000:2003]))
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
#   print("y_test %s X_test %s" % (y_test[0:2], X_test[0:2]))
#   print("y_train %s %s %s %s \nX_train %s %s %s %s" % (y_train[0:2], y_train[500:502], \
#          y_train[1000:1002], y_train[1500:1502], X_train[0:2], X_train[500:502], \
#          X_train[1000:1002], X_train[1500:1502]))
# indices seem ok

    model = sm.OLS(y_train, X_train)   # ordinary least squares
    f = model.fit()
#   print 'full model fit summary\n', f.summary()
    met = metric_summary(f)
    metrics.append(met)

metrics = zip(*metrics)  # list of tuples

print '\nraw metrics', metrics

# [mae, mad, mse, rsq]
print '\nmetrics: mean absolute error, median absolute error, mean squared error, r-squared'
print_metric_stats('mae', metrics[0])
print_metric_stats('mad', metrics[1])
print_metric_stats('mse', metrics[2])
print_metric_stats('rsq', metrics[3])

print "\nConclusions:\n  mean of each metric is same as full dataset within 3-4 decimal places"
print "    (if they're slightly lower, probably due to rounding error)"
print "  sd of each metric is 0.3-0.4% for mae & rsq, 0.77% for mad & mse"
print "  not too surprising that mse % sd > mae % sd, as mse errors are squared"
print "  mae and rsq are both measures of goodness-of-fit, but rsq is normalized to 1"
print "  mad may be more influenced by outliers than mae"
print "  which one is more reliable?  it depends what you're trying to measure"


