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

sown = list(set(loansData['Home.Ownership']))
def own_to_num(s):
    return sown.index(s)

slurp = list(set(loansData['Loan.Purpose']))
def purpose_to_num(s):
    return slurp.index(s)

loansData['Interest.Rate'] = loansData['Interest.Rate'].apply(lambda s: float(s.rstrip('%')))
loansData['Debt.To.Income.Ratio'] = loansData['Debt.To.Income.Ratio'].apply(lambda s: float(s.rstrip('%')))
loansData['IR_TF'] = loansData['Interest.Rate'].apply(lambda x: 0 if x<12 else 1)
loansData['Loan.Length'] = loansData['Loan.Length'].apply(lambda s: int(s.rstrip(' months')))
loansData['FICO.Score'] = loansData['FICO.Range'].apply(splitSum)
loansData['Home.Type'] = loansData['Home.Ownership'].apply(own_to_num)
loansData['Loan.Purpose.Score'] = loansData['Loan.Purpose'].apply(purpose_to_num)
loansData['Intercept'] = 1

print 'loansData head\n', loansData[:5]
print 'loansData describe\n', loansData.describe()

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

# plot predicted and incorrect target values
def plot_predict(label, correct, incorrect):
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
    plt.savefig(plotdir+label+'_bayes_simple_intrate_predict.png')

def plot_theo(label, correct, incorrect):
# plot theoretical predicted not target (IR_TF) values
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
    plt.savefig(plotdir+label+'_bayes_simple_intrate_theo.png')

def do_naive_bayes(indep_variables, label, predict_plot=False, theo_plot=False):
    print 'label:', label
    print 'Dependent Variable(s):', dep_variables
    print 'Independent Variables:', indep_variables

    loans_data = pd.DataFrame( loansData[indep_variables] )
#   print 'loans_data head\n', loans_data[:5]

    pred = gnb.fit(loans_data, loans_target).predict(loans_data)

    print(">>> Number of mislabeled points out of a total %d points : %d <<<" \
          % ( loans_data.shape[0], (loans_target != pred).sum() ))
    print "Number of correctly labeled predicted points : %d" % (loans_target == pred).sum()

    loans_data['target'] = loans_target
    loans_data['predict'] = pred
#   print 'loans_data head\n', loans_data[:5]

    incorrect = loans_data[ loans_data['target'] != loans_data['predict'] ]
    correct = loans_data[ loans_data['target'] == loans_data['predict'] ]
#   print 'loans_data incorrectly labeled head\n', incorrect[:5]

    if (predict_plot):
        plot_predict(label, correct, incorrect)

    if (theo_plot):
        plot_theo(label, correct, incorrect)

    return (loans_target != pred).sum()

indep_variables = ['FICO.Score', 'Amount.Requested']
do_naive_bayes(indep_variables, label='fa', predict_plot=True, theo_plot=True)

indep_variables = ['FICO.Score', 'Amount.Requested', 'Home.Type']
do_naive_bayes(indep_variables, label='fah',predict_plot=True)

indep_variables = ['FICO.Score', 'Amount.Requested', 'Home.Type', 'Revolving.CREDIT.Balance', 'Monthly.Income', 'Open.CREDIT.Lines', 'Debt.To.Income.Ratio']
do_naive_bayes(indep_variables, label='all7', predict_plot=True)

indep_variables = ['FICO.Score', 'Amount.Requested', 'Home.Type', 'Revolving.CREDIT.Balance', 'Monthly.Income', 'Open.CREDIT.Lines', 'Debt.To.Income.Ratio', 'Loan.Length', 'Loan.Purpose.Score', 'Amount.Funded.By.Investors', 'Inquiries.in.the.Last.6.Months']
do_naive_bayes(indep_variables, label='all', predict_plot=True)

indep_variables = ['FICO.Score', 'Amount.Requested', 'Home.Type', 'Loan.Length', 'Loan.Purpose.Score', 'Amount.Funded.By.Investors', 'Inquiries.in.the.Last.6.Months']
do_naive_bayes(indep_variables, label='better', predict_plot=True)

# to do:
#    plot expected IR_TF from logistic regression function, compare w/ naive bayes
#    use KFold to properly measure prediction (do not plot by default)
#    add automatic random combinations of variables to minimize incorrect number

print '\nplots created'


