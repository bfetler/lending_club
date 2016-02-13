# cross validation of linear regression

import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.cross_validation import KFold
from statsmodels.tools.eval_measures import meanabs
# from statsmodels.tools.eval_measures import mse, iqr
import matplotlib.pyplot as plt
import re
import os

# loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
loansData = pd.read_csv('data/loansData.csv')
loansData.dropna(inplace=True)

plotdir = 'linear_cross_plots/'
if not os.access(plotdir, os.F_OK):
    os.mkdir(plotdir)

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

def mean_abs_error1(y, predict):
    sum = 0.0
    for (yval, pval) in zip(y, predict):
        sum = sum + np.abs(yval - pval)
    return sum / len(y)

def mean_abs_error2(resid):
    sum = 0.0
    for rval in resid:
        sum = sum + np.abs(rval)
    return sum / len(resid)

def mean_abs_error3(y, predict):
#   return statsmodels.tools.eval_measures.meanabs(y, predict)
    return np.mean(meanabs(y, predict))

# seriously, there ought to be a better way
def median_abs_error1(y, predict):
    yy = y.flatten()
    yp = predict.reshape(yy.shape)
    return np.median(np.abs(yy, yp))

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
print 'full fit r-squared %g, rsq-adj %g, mse_resid %g, mse_total %g, ssr %g' % \
       (f.rsquared, f.rsquared_adj, f.mse_resid, f.mse_total, f.ssr)
predict = f.predict()
# print 'full fit meanabs1 %g, meanabs2 %g, meanabs3 %g' % \
#        (mean_abs_error1(y, predict), mean_abs_error2(f.resid), mean_abs_error3(y, predict))
print 'full fit meanabs1 %g' % (mean_abs_error1(y, predict))
print 'full fit meanabs2', mean_abs_error2(f.resid)
mae3 = mean_abs_error3(y, predict)
print 'full fit meanabs3', mae3.__class__, mae3
mad1 = median_abs_error1(y, predict)
print 'full fit medianabs1', mad1.__class__, mad1
print 'full fit p-values', f.pvalues
# print 'dir(fit)', dir(f)
# print 'f.resid class', f.resid.__class__
# mae1 = np.mean(meanabs(y, predict))
# mse1 = np.mean(mse(y, predict))
# iqr1 = np.mean(iqr(y, predict))
# mad1 = np.median(meanabs(y, predict))
# print 'full fit mae1', mae1.__class__, mae1
# print 'full fit mse1', mse1.__class__, mse1
# print 'full fit iqr1', iqr1.__class__, iqr1
# print 'full fit mad1', mad1.__class__, mad1

# yy = y.flatten()
# predict = predict.reshape(yy.shape)
# print 'y class, predict class', y.__class__, yy.__class__, yy.shape, predict.__class__, predict.shape
# mae2 = np.mean(np.abs(yy, predict))
# mad2 = np.median(np.abs(yy, predict))
# print 'full fit mae2', mae2.__class__, mae2
# print 'full fit mad2', mad2.__class__, mad2

# plot exploratory data
# plt.clf()
# loansData.groupby('Loan.Purpose').hist()     # more useful
# plt.savefig(plotdir+'LoanPurpose_Histogram.png')

# test KFold fitting
# split data into ten folds, repeat OLS, compute metrics on each one
print('len X %d, len y %d' % (len(X), len(y)) )



