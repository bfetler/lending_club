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
# really it should be '1 if x<12 else 0' if 1 True, 0 False since x<12 is happier outcome
loansData['Loan.Length'] = loansData['Loan.Length'].apply(lambda s: int(s.rstrip(' months')))
# loansData['FICO.Score'] = loansData['FICO.Range'].map(lambda s: int(s.split('-')[0]))
loansData['FICO.Score'] = loansData['FICO.Range'].apply(splitSum)
loansData['Intercept'] = 1

print 'loansData head\n', loansData[:5]
# print '\nloansData basic stats\n', loansData.describe()   # print basic stats

# print test of IR_TF calculation
# print 'loansData IntRate < 10\n', loansData[loansData['Interest.Rate'] < 10][:5]
# print 'loansData IntRate > 13\n', loansData[loansData['Interest.Rate'] > 13][:5]
# print 'loansData IntRate > 13\n', loansData[loansData['Interest.Rate'] > 13][:5]
print 'loansData IntRate < 10\n', loansData[loansData['Interest.Rate'] < 10][:5]
print 'loansData FICO > 820\n', loansData[loansData['FICO.Score'] > 820]
print 'loansData FICO < 650\n', loansData[loansData['FICO.Score'] < 650]


# logistic regression example
# Q: What is the probability of getting a loan from the Lending Club
#    for $10,000 at an interest rate <= 12% with a FICO score of 750?
# Model: log(p/1-p) = mx + b  "logit function"
#   invert to find p(x) = 1 / (1 + exp(mx + b))  "logistic function"
# Wiki: The logit function is the quantile function of the
#       logistic distribution, while the probit is the quantile function
#       of the normal distribution.  See https://en.wikipedia.org/wiki/Logit

# Probability isn't binary, *assume* p<70% won't get the loan.
#   if p >= 0.70 then 1, else 0
# Try to calculate Interest.Rate, a dependent variable?
# Loan Amount is Amount.Requested (see linear_regression.py).
# Amount.Funded.By.Investors is also dependent?
dep_variables = ['Interest.Rate', 'Amount.Funded.By.Investors']
indep_variables = ['FICO.Score', 'Amount.Requested', 'Loan.Length', 'Loan.Purpose', 'Debt.To.Income.Ratio', 'State', 'Home.Ownership', 'Monthly.Income', 'Open.CREDIT.Lines', 'Revolving.CREDIT.Balance', 'Inquiries.in.the.Last.6.Months', 'Employment.Length']

# Really what we want is:
#    -using linear regression eq, plug in values
#    -calculate probability then p <= 12%
# interest_rate = b + a1 * FICO.Score + a2 * Amount.Requested
#               = b + a1 * 750 + a2 * 10000
print '''interest_rate = b + a1 * FICO.Score + a2 * Amount.Requested
              = b + a1 * 750 + a2 * 10000'''
print 'find p(x) = 1 / (1 + exp(a1*x1 + a2*x2 + b))  "logistic function"'

dep_variables = ['IR_TF']
indep_variables = ['FICO.Score', 'Amount.Requested', 'Intercept']
print 'Dependent Variable(s):', dep_variables
print 'Independent Variables:', indep_variables

logit = sm.Logit( loansData['IR_TF'], loansData[indep_variables] )
result = logit.fit()
print 'fit coefficients:\n', result.params

# Why do my coefficients have opposite sign of thinkful notes?
# Because IR_TF lambda expression is backwards?

def logistic_fn_orig(loanAmount, fico, params):
#   a1 = -0.0875    # approximate hard-coded values to start
#   a2 =  0.000174
#   b  =  60.347
#   print 'params', # params.__class__,  params  # class Series
    a1 = params['FICO.Score']
    a2 = params['Amount.Requested']
    b  = params['Intercept']
    p  = 1 / (1 + np.exp( b + a1 * fico + a2 * loanAmount ))
    return p

def logistic_fn(loanAmount, fico, params):
    a1 = -params['FICO.Score']
    a2 = -params['Amount.Requested']
    b  = -params['Intercept']
    p  = 1 / (1 + np.exp( b + a1 * fico + a2 * loanAmount ))
    return p

def pred_orig(loanAmount, fico, params):
    msg = '  You will '
    p = logistic_fn(loanAmount, fico, params)
    if float(p) < 0.7:
#   if float(p) < 0.3:  # if IR_TF backwards, compare to 1.0 - 0.7 = 0.3?
        msg += 'NOT '
    msg += 'get the loan for under 12 percent.'
    return msg

def pred(loanAmount, fico, params):
    msg = '  You will '
    p = logistic_fn(loanAmount, fico, params)
#   if float(p) > 0.7:
    if float(p) > 0.3:  # if IR_TF backwards, compare to 1.0 - 0.7 = 0.3?
        msg += 'NOT '
    msg += 'get the loan for under 12 percent.'
    return msg

print 'logistic values:\nloan  fico probability'
print 10000, 750, logistic_fn(10000, 750, result.params), pred(10000, 750, result.params)
print 10000, 720, logistic_fn(10000, 720, result.params), pred(10000, 720, result.params)
print 10000, 710, logistic_fn(10000, 710, result.params), pred(10000, 710, result.params)
print 10000, 700, logistic_fn(10000, 700, result.params), pred(10000, 700, result.params)
print 10000, 690, logistic_fn(10000, 690, result.params), pred(10000, 690, result.params)

print '\nThe probability that we can obtain a loan at less than 12 percent interest for $10000 USD with a FICO score of 720 is: %.1f percent.  It is more likely than not we will get the loan for under 12 percent.' % ( 100 - 100 * logistic_fn(10000, 720, result.params) )

plt.clf()
fico_array = range(540, 860, 10)
fico_logit = map(lambda x: logistic_fn(10000, x, result.params), fico_array)
# print 'fico array:', fico_array, fico_logit
plt.plot(fico_array, fico_logit)
plt.xlim(550, 850)
plt.xlabel('FICO Score')
plt.ylabel('Probability : Interest Rate > 12%')
plt.title('Logistic Plot')
plt.text(590, 0.25, ' Lower FICO Score ~\nHigher Interest Rate')
plt.savefig(plotdir+'fico_logistic.png')

plt.clf()
divvy = 20
loan_array = map(lambda x: 10 ** (float(x) / divvy), range(2*divvy, 5*divvy))
loan_logit = map(lambda x: logistic_fn(x, 720, result.params), loan_array)
# print 'loan array:', loan_array, loan_logit
plt.plot(loan_array, loan_logit)
# plt.xscale('log')
plt.xlim(0, 40000)
locs, labels = plt.xticks()
plt.xticks(locs, map(lambda x: '$'+str(int(x/1000))+'k', locs))
plt.xlabel('Loan Amount Requested, USD')
plt.ylabel('Probability : Interest Rate > 12%')
plt.title('Logistic Plot')
plt.text(14000, 0.25, 'Higher Loan Amount ~ Lower FICO Score\n          ~ Higher Interest Rate')
plt.savefig(plotdir+'loan_logistic.png')

plt.clf()
# plt.plot(loansData['FICO.Score'], loansData['Amount.Requested'], 'o', color='#ff00ff')
plt.scatter(loansData['FICO.Score'], loansData['Amount.Requested'], c=loansData['IR_TF'], linewidths=0)
plt.xlim(620, 850)
plt.ylim(0, 40000)
locs, labels = plt.yticks()
plt.yticks(locs, map(lambda x: '$'+str(int(x/1000))+'k', locs))
plt.xlabel('FICO Score')
plt.ylabel('Loan Amount Requested, USD')
plt.title('Interest Rates: red > 12%, blue < 12%')
# plt.legend(['red > 12% interest, blue < 12% interest'])
plt.savefig(plotdir+'loan_v_fico.png')

print '\nplots created: fico_logistic.png, loan_logistic.png, loan_v_fico.png'

