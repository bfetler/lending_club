# unit 2.3 linear regression

import pandas as pd
import re
import matplotlib.pyplot as plt

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
loansData.dropna(inplace=True)

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
# loansData['FICO.Average'] = loansData['FICO.Range'].map(splitSum)
# loansData['FICO.Score'] = [int(val.split('-')[0]) for val in loansData['FICO.Range']]
# apply and map both work, map may be more FP standard
# may also use list comprehension

print 'loansData head\n', loansData[:5]
print '\nloansData basic stats\n', loansData.describe()   # print basic stats
# loansData.boxplot()  # not too useful, scales are very different
# loansData.hist()     # more useful
# loansData.groupby('LoanPurpose').hist()     # more useful
# plt.show()

# plt.clf()
# loansData['FICO.Score'].hist()
# plt.show()

# pd.scatter_matrix(loansData, alpha=0.05, figsize=(10,10))
# diagonal='hist' by default, see help(pd.scatter_matrix)
# but some docs describe diagonal='kde' as default
pd.scatter_matrix(loansData, alpha=0.05, figsize=(10,10), diagonal='hist')
plt.show()


