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
    t = re.findall(pat, s)[0]
    return (int(t[0]) + int(t[1])) / 2

def own_to_num(s):
    if s == 'RENT':
        return 1
    elif s == 'MORTGAGE':
        return 2
    elif s == 'OWN':
        return 3
    else:   # 'ANY'
        return 0

loansData['Interest.Rate'] = loansData['Interest.Rate'].apply(lambda s: float(s.rstrip('%')))
loansData['Debt.To.Income.Ratio'] = loansData['Debt.To.Income.Ratio'].apply(lambda s: float(s.rstrip('%')))
loansData['IR_TF'] = loansData['Interest.Rate'].apply(lambda x: 0 if x<12 else 1)
loansData['Loan.Length'] = loansData['Loan.Length'].apply(lambda s: int(s.rstrip(' months')))
loansData['FICO.Score'] = loansData['FICO.Range'].apply(splitSum)
loansData['Home.Type'] = loansData['Home.Ownership'].apply(own_to_num)
loansData['Intercept'] = 1

print 'loansData head\n', loansData[:5]

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

gnb = GaussianNB()
dep_variables = ['IR_TF']
loans_target = loansData['IR_TF']
print 'loans_target head\n', loans_target[:5]

def plot_target(label, correct, incorrect):
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
    plt.savefig(plotdir+label+'_bayes_simple_intrate_incorrect.png')

def plot_predicted(label, correct, incorrect):
# plot predicted not target (IR_TF) values
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
    plt.savefig(plotdir+label+'_bayes_simple_intrate_predicted.png')

def do_naive_bayes(indep_variables, label, target_plot=True, pred_plot=False):
    print 'label:', label
    print 'Dependent Variable(s):', dep_variables
    print 'Independent Variables:', indep_variables

    loans_data = pd.DataFrame( loansData[indep_variables] )
    print 'loans_data head\n', loans_data[:5]

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

    if (target_plot):
        plot_target(label, correct, incorrect)

    if (pred_plot):
        plot_predicted(label, correct, incorrect)

    return (loans_target != pred).sum()

indep_variables = ['FICO.Score', 'Amount.Requested']
do_naive_bayes(indep_variables, label='fa', pred_plot=True)

indep_variables = ['FICO.Score', 'Amount.Requested', 'Home.Type']
do_naive_bayes(indep_variables, label='fah')

indep_variables = ['FICO.Score', 'Amount.Requested', 'Home.Type', 'Revolving.CREDIT.Balance', 'Monthly.Income', 'Open.CREDIT.Lines', 'Debt.To.Income.Ratio']
do_naive_bayes(indep_variables, label='all')

# to do:
#    use KFold to properly measure prediction (do not plot by default)
#    add combinations of variables automatically to minimize incorrect

print '\nplots created'


#### skip from here
## plot expected IR_TF from logistic regression function?  compare w/ naive bayes

# logistic regression:
# Find probability of getting a loan from the Lending Club for
#   $10,000 at an interest rate <= 12% with a FICO score of 750.
#   logistic function p(x) = 1 / (1 + exp(mx + b))

# Probability isn't binary, *assume* p<70% won't get the loan.
#   if p >= 0.70 then 1, else 0

# We don't actually need any of the sm.Logit.fit() calculations from previous exercise.
# We do need IR_TF column as Naive Bayes target.

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

