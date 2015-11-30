# unit 2.3 linear regression

import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import re
import os

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
loansData.dropna(inplace=True)

plotdir = 'l_regression/'
if not os.access(plotdir, os.F_OK):
    os.mkdir(plotdir)

pat = re.compile('(.*)-(.*)')  # ()'s return two matching fields

def splitSum(s):
#   t = s.split('-')
    t = re.findall(pat, s)[0]
    return (int(t[0]) + int(t[1])) / 2

loansData['Interest.Rate'] = loansData['Interest.Rate'].apply(lambda s: float(s.rstrip('%')))
loansData['Loan.Length'] = loansData['Loan.Length'].apply(lambda s: int(s.rstrip(' months')))
# loansData['FICO.Score'] = loansData['FICO.Range'].apply(lambda s: int(s.split('-')[0]))
loansData['FICO.Average'] = loansData['FICO.Range'].apply(splitSum)
#   apply and map both work, map more FP standard?
# loansData['FICO.Average'] = loansData['FICO.Range'].map(splitSum)
#   may also use list comprehension
# loansData['FICO.Score'] = [int(val.split('-')[0]) for val in loansData['FICO.Range']]

print 'loansData head\n', loansData[:5]
print '\nloansData basic stats\n', loansData.describe()   # print basic stats


# calculate linear regression example
# equation: InterestRate = b + a1 * FICO.Average + a2 * Loan.Amount
y  = np.matrix(loansData['Interest.Rate']).transpose()
x1 = np.matrix(loansData['Amount.Requested']).T
x2 = np.matrix(loansData['FICO.Average']).T
# x2 = np.matrix(loansData['Loan.Length']).T  # try different variable, worse fit: R^2 ~ 0.21, p-values still zero
print 'IntRate matrix', y[:5]
print 'Amt matrix', x1[:5]
print 'FICO matrix', x2[:5]
# x = np.column_stack([x1, x2])
X = sm.add_constant( np.column_stack([x1, x2]) )  # x's plus constant
model = sm.OLS(y, X)
f = model.fit()
print 'model fit summary\n', f.summary()
print 'fit r-squared', f.rsquared, ', p-values', f.pvalues
# rsquared 0.657, pvalues < 1e-203
# pvalues are all very close to zero, <= 0.05, fits H0

# what does it mean?
# print 'fit methods', dir(f)
# dir(sm) help(sm)
# help(sm.add_constant) => add a column of ones
# help(sm.OLS) => OLS Ordinary Least Squares Regression
# fit uses method='pinv' pseudo-inverse by default


# plot exploratory data
plt.clf()
# loansData.boxplot()  # not too useful, scales are very different
# loansData.hist()     # more useful
loansData.groupby('Loan.Purpose').hist()     # more useful
plt.savefig(plotdir+'LoanPurpose_Histogram.png')

plt.clf()
loansData['FICO.Average'].hist()
plt.savefig(plotdir+'FICO_Histogram.png')

plt.clf()
# pd.scatter_matrix(loansData, alpha=0.05, figsize=(10,10))
# diagonal='hist' by default, see help(pd.scatter_matrix)
# but some docs describe diagonal='kde' as default
pd.scatter_matrix(loansData, alpha=0.05, figsize=(10,10), diagonal='hist')
plt.savefig(plotdir+'scatter_matrix.png')


