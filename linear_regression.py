# unit 2.3 linear regression

import pandas as pd

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

print loansData[:5]

loansData['Interest.Rate.Pct'] = loansData['Interest.Rate'].apply(lambda s: s.rstrip('%'))
loansData['Loan.Length.Mo'] = loansData['Loan.Length'].apply(lambda s: s.rstrip(' months'))

print loansData[:5]

