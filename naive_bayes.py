# naive bayes, classes based on logistic regression

import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import re
import os

# loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
loansData = pd.read_csv('data/loansData.csv')  # downloaded data if no internet
loansData.dropna(inplace=True)

plotdir = 'naive_bayes_plots/'
if not os.access(plotdir, os.F_OK):
    os.mkdir(plotdir)

pat = re.compile('(.*)-(.*)')  # ()'s return two matching fields

def splitSum(s):
#   t = s.split('-')
    t = re.findall(pat, s)[0]
    return (int(t[0]) + int(t[1])) / 2

loansData['Interest.Rate'] = loansData['Interest.Rate'].apply(lambda s: float(s.rstrip('%')))
loansData['IR_TF'] = loansData['Interest.Rate'].apply(lambda x: 0 if x<12 else 1)
loansData['Loan.Length'] = loansData['Loan.Length'].apply(lambda s: int(s.rstrip(' months')))
loansData['FICO.Score'] = loansData['FICO.Range'].apply(splitSum)
loansData['Intercept'] = 1

print 'loansData head\n', loansData[:5]

# print 'loansData IntRate > 13\n', loansData[loansData['Interest.Rate'] > 13][:5]
# print 'loansData IntRate < 10\n', loansData[loansData['Interest.Rate'] < 10][:5]
# print 'loansData FICO > 820\n', loansData[loansData['FICO.Score'] > 820]
# print 'loansData FICO < 650\n', loansData[loansData['FICO.Score'] < 650]

# logistic regression:
# Find probability of getting a loan from the Lending Club for
#   $10,000 at an interest rate <= 12% with a FICO score of 750.
#   logistic function p(x) = 1 / (1 + exp(mx + b))

# Probability isn't binary, *assume* p<70% won't get the loan.
#   if p >= 0.70 then 1, else 0

# We don't actually need any of the sm.Logit.fit() calculations from previous exercise.
# We do need IR_TF column as Naive Bayes target.

#### skip from here

dep_variables = ['IR_TF']
indep_variables = ['FICO.Score', 'Amount.Requested']
print 'Dependent Variable(s):', dep_variables
print 'Independent Variables:', indep_variables

loans_data = pd.DataFrame( loansData[indep_variables] )
loans_target = loansData['IR_TF']
print 'loans_data head\n', loans_data[:5]
print 'loans_target head\n', loans_target[:5]

gnb = GaussianNB()
pred = gnb.fit(loans_data, loans_target).predict(loans_data)

print("Number of mislabeled points out of a total %d points : %d" \
      % ( loans_data.shape[0], (loans_target != pred).sum() ))
print "pred correctly labeled", (loans_target == pred).sum()

loans_data['target'] = loans_target
loans_data['predict'] = pred
print 'loans_data head\n', loans_data[:5]

incorrect = loans_data[ loans_data['target'] != loans_data['predict'] ]
correct = loans_data[ loans_data['target'] == loans_data['predict'] ]
print 'loans_data incorrectly labeled head\n', incorrect[:5]

# need predicted not target (IR_TF) values
loans_data = pd.DataFrame( loansData[indep_variables] )
loans_target = loansData['IR_TF']

plt.clf()
plt.scatter(loansData['FICO.Score'], loansData['Amount.Requested'], c=loansData['IR_TF'], linewidths=0)
plt.xlim(620, 850)
plt.ylim(0, 40000)
locs, labels = plt.yticks()
plt.yticks(locs, map(lambda x: '$'+str(int(x/1000))+'k', locs))
plt.xlabel('FICO Score')
plt.ylabel('Loan Amount Requested, USD')
plt.title('Target Interest Rates: red > 12%, blue < 12%')
plt.savefig(plotdir+'intrate_target.png')

plt.clf()
plt.scatter(correct['FICO.Score'], correct['Amount.Requested'], c=correct['target'], \
     linewidths=0)
plt.scatter(incorrect['FICO.Score'], incorrect['Amount.Requested'], c=incorrect['target'], \
     linewidths=1, s=20, marker='x')
plt.xlim(620, 850)
plt.ylim(0, 40000)
locs, labels = plt.yticks()
plt.yticks(locs, map(lambda x: '$'+str(int(x/1000))+'k', locs))
plt.xlabel('FICO Score')
plt.ylabel('Loan Amount Requested, USD')
plt.title('Naive Bayes Predicted Interest Rates: red > 12%, blue < 12%')
plt.savefig(plotdir+'bayes_simple_intrate_incorrect.png')

plt.clf()
plt.scatter(correct['FICO.Score'], correct['Amount.Requested'], c=correct['target'], \
     linewidths=0)
plt.scatter(incorrect['FICO.Score'], incorrect['Amount.Requested'], c=incorrect['predict'], \
     linewidths=1, s=20, marker='x')
plt.xlim(620, 850)
plt.ylim(0, 40000)
locs, labels = plt.yticks()
plt.yticks(locs, map(lambda x: '$'+str(int(x/1000))+'k', locs))
plt.xlabel('FICO Score')
plt.ylabel('Loan Amount Requested, USD')
plt.title('Naive Bayes Predicted Interest Rates: red > 12%, blue < 12%')
plt.savefig(plotdir+'bayes_simple_intrate_predicted.png')

# print '\nplots created: fico_logistic.png, loan_logistic.png, loan_v_fico.png'

#### skip from here
## plot expected IR_TF from logistic regression function?  compare w/ naive bayes

print ' \nSKIP FROM HERE'
print '''interest_rate = b + a1 * FICO.Score + a2 * Amount.Requested
              = b + a1 * 750 + a2 * 10000'''
print 'find p(x) = 1 / (1 + exp(a1*x1 + a2*x2 + b))  "logistic function"'

dep_variables = ['IR_TF']
indep_variables = ['FICO.Score', 'Amount.Requested', 'Intercept']

logit = sm.Logit( loansData['IR_TF'], loansData[indep_variables] )
result = logit.fit()
print 'fit coefficients:\n', result.params

# Why do my coefficients have opposite sign of thinkful notes?
# Because IR_TF lambda expression is backwards?

def logistic_fn(loanAmount, fico, params):
    a1 = -params['FICO.Score']
    a2 = -params['Amount.Requested']
    b  = -params['Intercept']
    p  = 1 / (1 + np.exp( b + a1 * fico + a2 * loanAmount ))
    return p

def pred(loanAmount, fico, params):
    msg = '  You will '
    p = logistic_fn(loanAmount, fico, params)
    if float(p) > 0.3:  # if IR_TF backwards, compare to 1.0 - 0.7 = 0.3
        msg += 'NOT '
    msg += 'get the loan for under 12 percent.'
    return msg

print 'logistic values:\nloan  fico probability'
print 10000, 750, logistic_fn(10000, 750, result.params), pred(10000, 750, result.params)
print 10000, 720, logistic_fn(10000, 720, result.params), pred(10000, 720, result.params)
print 10000, 710, logistic_fn(10000, 710, result.params), pred(10000, 710, result.params)
print 10000, 700, logistic_fn(10000, 700, result.params), pred(10000, 700, result.params)
print 10000, 690, logistic_fn(10000, 690, result.params), pred(10000, 690, result.params)

#### skip to here

