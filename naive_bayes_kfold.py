# naive bayes with cross validation, classes based on logistic regression

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
import re
import os


# to do:
#    plot expected IR_TF from logistic regression function, compare w/ naive bayes
#    add automatic random combinations of variables to minimize incorrect number

# loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
loansData = pd.read_csv('data/loansData.csv')  # downloaded data if no internet
loansData.dropna(inplace=True)

plotdir = 'naive_bayes_kfold_plots/'
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
loansData['IR_TF'] = loansData['Interest.Rate'].apply(lambda x: 0 if x<12 else 1)
loansData['Debt.To.Income.Ratio'] = loansData['Debt.To.Income.Ratio'].apply(lambda s: float(s.rstrip('%')))
loansData['Loan.Length'] = loansData['Loan.Length'].apply(lambda s: int(s.rstrip(' months')))
loansData['FICO.Score'] = loansData['FICO.Range'].apply(splitSum)
loansData['Home.Type'] = loansData['Home.Ownership'].apply(own_to_num)
loansData['Loan.Purpose.Score'] = loansData['Loan.Purpose'].apply(purpose_to_num)

print 'loansData head\n', loansData[:5]
print 'loansData describe\n', loansData.describe()

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

def naive_bayes_fold(train_data, train_target, test_data, test_target):
    pred = gnb.fit(train_data, train_target).predict(test_data)
#   print "Number of incorrectly labeled predicted points : %d out of %d" \
#         % ( (test_target != pred).sum(), test_target.shape[0] )
#   print "Number of correctly labeled predicted points : %d" % (test_target == pred).sum()
    return (test_target != pred).sum()

def do_naive_bayes(indep_variables, label, predict_plot=False, theo_plot=False):
    print 'label:', label
    print 'Dependent Variable(s):', dep_variables
    print 'Independent Variables:', indep_variables

# do it all with pd.DataFrame (could also do with np.ndarray)
    loans_data = pd.DataFrame( loansData[indep_variables] )

    metrics = []
    kf = KFold(loans_data.shape[0], n_folds=4)
    for train, test in kf:
#       print("test (len %d) %s %s, train (len %d) %s %s %s %s" % (len(test), test[:3], \
#          test[-3:], len(train), train[:3], train[624:626], train[1250:1253], train[-3:] ))
        train_data, test_data, train_target, test_target = loans_data.iloc[train], loans_data.iloc[test], loans_target.iloc[train], loans_target.iloc[test]
#       print("shape train_data %s, test_data %s, train_target %s, test_target %s" % \
#          (train_data.shape, test_data.shape, train_target.shape, test_target.shape ))
#       print("train_target %s %s\ntrain_data %s %s %s %s" % ( train_target[:3], \
#           train_target[-3:], train_data[:3], train_data[624:626], train_data[1250:1253], \
#           train_data[-3:] ))
        met = naive_bayes_fold(train_data, train_target, test_data, test_target)
        metrics.append(met)

    score = reduce(lambda x,y: x + y, metrics)
    print "score: number of incorrectly labeled points: %d out of %d " % \
         ( score, loans_target.shape[0] )

#   if (predict_plot):
#       plot_predict(label, correct, incorrect)

#   if (theo_plot):
#       plot_theo(label, correct, incorrect)

    return score

indep_variables = ['FICO.Score', 'Amount.Requested']
do_naive_bayes(indep_variables, label='fa', predict_plot=True)

indep_variables = ['FICO.Score', 'Amount.Requested', 'Home.Type']
do_naive_bayes(indep_variables, label='fah', predict_plot=True)

indep_variables = ['FICO.Score', 'Amount.Requested', 'Home.Type', 'Revolving.CREDIT.Balance', 'Monthly.Income', 'Open.CREDIT.Lines', 'Debt.To.Income.Ratio']
do_naive_bayes(indep_variables, label='all7', predict_plot=True)

indep_variables = ['FICO.Score', 'Amount.Requested', 'Home.Type', 'Revolving.CREDIT.Balance', 'Monthly.Income', 'Open.CREDIT.Lines', 'Debt.To.Income.Ratio', 'Loan.Length', 'Loan.Purpose.Score', 'Amount.Funded.By.Investors', 'Inquiries.in.the.Last.6.Months']
do_naive_bayes(indep_variables, label='all', predict_plot=True)

indep_variables = ['FICO.Score', 'Amount.Requested', 'Home.Type', 'Loan.Length', 'Loan.Purpose.Score', 'Amount.Funded.By.Investors', 'Inquiries.in.the.Last.6.Months']
do_naive_bayes(indep_variables, label='better', predict_plot=True)


# print '\nplots created'


