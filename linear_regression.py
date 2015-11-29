# unit 2.3 linear regression

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import re
import os

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
loansData.dropna(inplace=True)

plotdir = 'l_regression/'
if not os.access(plotdir, os.F_OK):
    os.mkdir(plotdir)

# print loansData[:5]
pat = re.compile('(.*)-(.*)')  # ()'s return two matching fields

def splitSum(s):
#   t = s.split('-')
#   return (int(t[0]) + int(t[1])) / 2
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


