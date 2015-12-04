# unit 2.4 logistic regression

import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import re
import os

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
# write out loansData.to_csv(...) only if continuing on from linear_regression.py
loansData.dropna(inplace=True)

plotdir = 'logistic_plots/'
if not os.access(plotdir, os.F_OK):
    os.mkdir(plotdir)

pat = re.compile('(.*)-(.*)')  # ()'s return two matching fields

def splitSum(s):
#   t = s.split('-')
    t = re.findall(pat, s)[0]
    return (int(t[0]) + int(t[1])) / 2

loansData['Interest.Rate'] = loansData['Interest.Rate'].apply(lambda s: float(s.rstrip('%')))
loansData['IR_TF'] = loansData['Interest.Rate'].apply(lambda x: 0 if x<12 else 1)
# python ternary if syntax is goofy
loansData['Loan.Length'] = loansData['Loan.Length'].apply(lambda s: int(s.rstrip(' months')))
# loansData['FICO.Score'] = loansData['FICO.Range'].map(lambda s: int(s.split('-')[0]))
loansData['FICO.Score'] = loansData['FICO.Range'].apply(splitSum)
loansData['Logic.Intercept'] = 1

print 'loansData head\n', loansData[:5]
# print '\nloansData basic stats\n', loansData.describe()   # print basic stats

# print basic test of IR_TF calculation
print 'loansData IntRate == 10\n', loansData[loansData['Interest.Rate'] == 10][:5]
print 'loansData IntRate < 10\n', loansData[loansData['Interest.Rate'] < 10][:5]
print 'loansData IntRate > 13\n', loansData[loansData['Interest.Rate'] > 13][:5]


# logistic regression example
# Q: What is the probability of getting a loan from the Lending Club
#    for $10,000 at an interest rate <= 12% with a FICO score of 750?
# Model: log(p/1-p) = mx + b
# Try to calculate Interest.Rate, dependent variable?
# Amount.Funded.By.Investors may also be dependent?
dep_variables = ['Interest.Rate', 'Amount.Funded.By.Investors']
indep_variables = ['FICO.Score', 'Amount.Requested', 'Loan.Length', 'Loan.Purpose', 'Debt.To.Income.Ratio', 'State', 'Home.Ownership', 'Monthly.Income', 'Open.CREDIT.Lines', 'Revolving.CREDIT.Balance', 'Inquiries.in.the.Last.6.Months', 'Employment.Length']
print 'Dependent Variable(s):', dep_variables
print 'Independent Variables:', indep_variables


