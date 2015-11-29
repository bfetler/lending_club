# unit 2.3 linear regression

import pandas as pd
import re
import matplotlib.pyplot as plt

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

# print loansData[:5]
pat = re.compile('(.*)(-)(.*)')

def splitSum(s):
#   t = s.split('-')
#   return (int(t[0]) + int(t[1])) / 2
    t = re.findall(pat, s)[0]
    return (int(t[0]) + int(t[2])) / 2

loansData['Interest.Rate.Pct'] = loansData['Interest.Rate'].apply(lambda s: float(s.rstrip('%')))
loansData['Loan.Length.Mo'] = loansData['Loan.Length'].apply(lambda s: int(s.rstrip(' months')))
loansData['FICO.Score'] = loansData['FICO.Range'].apply(lambda s: int(s.split('-')[0]))
loansData['FICO.Average'] = loansData['FICO.Range'].apply(splitSum)

print loansData[:5]

plt.clf()
loansData['FICO.Score'].hist()
plt.show()

