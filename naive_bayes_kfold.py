# naive bayes with cross validation, classes based on logistic regression

import numpy as np
import numpy.random as rnd
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
from functools import reduce
import re
import os


# to do:
#    add k-fold cross-validation    (done)
#    add automatic random combinations of variables to minimize incorrect number  (done)
#    better random optimization is possible using marginal difference of adding variables?
#    plot expected IR_TF from logistic regression function, compare w/ naive bayes

def init_data():
    '''Initialize data.  Read data and process dataframe.'''
#   loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
    loansData = pd.read_csv('data/loansData.csv')  # downloaded data if no internet
    loansData.dropna(inplace=True)

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

#   classify Interest Rate as boolean True/False (IR_TF) as low if < 12%, high if >= 12%
    loansData['Interest.Rate'] = loansData['Interest.Rate'].apply(lambda s: float(s.rstrip('%')))
    loansData['IR_TF'] = loansData['Interest.Rate'].apply(lambda x: 0 if x<12 else 1)
    loansData['Debt.To.Income.Ratio'] = loansData['Debt.To.Income.Ratio'].apply(lambda s: float(s.rstrip('%')))
    loansData['Loan.Length'] = loansData['Loan.Length'].apply(lambda s: int(s.rstrip(' months')))
    loansData['FICO.Score'] = loansData['FICO.Range'].apply(splitSum)
    loansData['Home.Type'] = loansData['Home.Ownership'].apply(own_to_num)
    loansData['Loan.Purpose.Score'] = loansData['Loan.Purpose'].apply(purpose_to_num)

    print('loansData head\n', loansData[:5])
    print('loansData describe\n', loansData.describe())

    return loansData

def getVarStr(indep_vars):
    lineLength = 80
    vars = list(indep_vars)
    sw = ["Variables: ["]
    last = vars[-1]
    vars = list(map((lambda s: s + ","), vars))
    vars[-1] = last
    ix = 0
    for s in vars:
        if len(sw[ix]) + len(s) + 1 > lineLength:
            ix += 1
            sw.append("    ")
        sw[ix] += s
        if s != last:
            sw[ix] += " "
    sw[ix] += "]"
    varstr = reduce( (lambda a,b: a + "\n" + b), sw)
    return varstr, len(sw)

def make_plotdir(plotdir):
    if not os.access(plotdir, os.F_OK):
        os.mkdir(plotdir)
    return plotdir
    
def plot_predict(label, score, indep_variables, correct, incorrect):
    '''Plot predicted (correct and incorrect) target values.'''
    plt.clf()
    plt.scatter(correct['FICO.Score'], correct['Amount.Requested'], c=correct['target'], \
         linewidths=0)
    plt.scatter(incorrect['FICO.Score'], incorrect['Amount.Requested'], c=incorrect['target'], \
         linewidths=1, s=20, marker='x')
    plt.xlim(620, 850)
    plt.ylim(0, 45000)
    locs, labels = plt.yticks()
    plt.yticks(locs, map(lambda x: '$'+str(int(x/1000))+'k', locs))
    plt.xlabel('FICO Score')
    plt.ylabel('Loan Amount Requested, USD')
    plt.title('Naive Bayes K-Fold Predicted Interest Rate Class')
    sc = 100 * float(score) / loans_target.shape[0]
    txt = "Score: %.2f%% incorrect (%d x pts)" % (sc, score)
    plt.text(630, 42000, txt)
    plt.text(770, 42000, 'red > 12%, blue < 12%', bbox=dict(edgecolor='black', fill=False))
    txt, pos = getVarStr(indep_variables)
    plt.text(630, 38000 + 1500*(2-pos), txt, fontsize=10)
    plt.savefig(plotdir+label+'_bayes_intrate_predict.png')

def plot_theo(label, score, indep_variables, correct, incorrect):
    '''Plot theoretical predicted (not target IR_TF) values.'''
    plt.clf()
    plt.scatter(correct['FICO.Score'], correct['Amount.Requested'], c=correct['target'], \
         linewidths=0)
    plt.scatter(incorrect['FICO.Score'], incorrect['Amount.Requested'], c=incorrect['predict'], \
         linewidths=1, s=20, marker='x')
    plt.xlim(620, 850)
    plt.ylim(0, 45000)
    locs, labels = plt.yticks()
    plt.yticks(locs, map(lambda x: '$'+str(int(x/1000))+'k', locs))
    plt.xlabel('FICO Score')
    plt.ylabel('Loan Amount Requested, USD')
    plt.title('Naive Bayes K-Fold Theoretical Predicted Interest Rate Class')
    sc = 100 * float(score) / loans_target.shape[0]
    txt = "Score: %.2f%% incorrect (%d x pts)" % (sc, score)
    plt.text(630, 42000, txt)
    plt.text(770, 42000, 'red > 12%, blue < 12%', bbox=dict(edgecolor='black', fill=False))
    txt, pos = getVarStr(indep_variables)
    plt.text(630, 38000 + 1500*(2-pos), txt, fontsize=10)
    plt.savefig(plotdir+label+'_bayes_intrate_theo.png')

def naive_bayes_fold(train_data, train_target, test_data):
    '''Do naive bayes on train and test data.'''
    pred = gnb.fit(train_data, train_target).predict(test_data)
    return pred

def do_naive_bayes(indep_variables, label='_label', predict_plot=False, theo_plot=False):
    '''Do naive bayes prediction on list of independent variables.
       Use k-fold cross validation to validate test data.'''
    if (label != '_label'):
        print('label:', label)
        print('Dependent Variable(s):', dep_variables)
        print('Independent Variables:', indep_variables)

#   use pd.DataFrame (could also use np.ndarray)
    loans_data = pd.DataFrame( loansData[indep_variables] )

    pred = []
    kf = KFold(loans_data.shape[0], n_folds=4)
    for train, test in kf:
        train_data, test_data, train_target, test_target = loans_data.iloc[train], loans_data.iloc[test], loans_target.iloc[train], loans_target.iloc[test]
        pred_fold = naive_bayes_fold(train_data, train_target, test_data)
        pred.extend( pred_fold )

    loans_data['target'] = loans_target
    loans_data['predict'] = pred
    score = (loans_target != pred).sum()

    incorrect = loans_data[ loans_data['target'] != loans_data['predict'] ]
    correct = loans_data[ loans_data['target'] == loans_data['predict'] ]

    if (predict_plot):
        print("score: number of incorrectly labeled points: %d out of %d (%.2f percent)" % \
             ( score, loans_target.shape[0], 100 * float(score) / loans_target.shape[0] ))
        plot_predict(label, score, indep_variables, correct, incorrect)

    if (theo_plot):
        plot_theo(label, score, indep_variables, correct, incorrect)

    return score

def naive_bayes_tests():
    '''Test series of different independent variables, compare score.'''
    indep_variables = ['FICO.Score', 'Amount.Requested']
    do_naive_bayes(indep_variables, label='fa', predict_plot=True, theo_plot=True)

    indep_variables = ['FICO.Score', 'Amount.Requested', 'Home.Type']
    do_naive_bayes(indep_variables, label='fah')

    indep_variables = ['FICO.Score', 'Amount.Requested', 'Home.Type', 'Revolving.CREDIT.Balance', 'Monthly.Income', 'Open.CREDIT.Lines', 'Debt.To.Income.Ratio']
    do_naive_bayes(indep_variables, label='all7')

    indep_variables = ['FICO.Score', 'Amount.Requested', 'Home.Type', 'Revolving.CREDIT.Balance', 'Monthly.Income', 'Open.CREDIT.Lines', 'Debt.To.Income.Ratio', 'Loan.Length', 'Loan.Purpose.Score', 'Amount.Funded.By.Investors', 'Inquiries.in.the.Last.6.Months']
    do_naive_bayes(indep_variables, label='all')

    indep_variables = ['FICO.Score', 'Amount.Requested', 'Home.Type', 'Loan.Length', 'Loan.Purpose.Score', 'Amount.Funded.By.Investors', 'Inquiries.in.the.Last.6.Months']
    do_naive_bayes(indep_variables, label='better')


def random_opt(varlist, init_list):
    '''Optimize list by randomly adding variables,
       accept if score decreases to find local minimum.'''

    vlist = list(init_list)
    score = do_naive_bayes(vlist)
    offset = len(vlist)  # offset by length of initial vlist
    indices = list(range(len(varlist) - offset))
    rnd.shuffle(indices)
    for ix in indices:
        ilist  = list(vlist)
        ilist.append(varlist[ix + offset])
        iscore = do_naive_bayes(ilist)
        if iscore < score:
            vlist = list(ilist)
            score = iscore

    print(">>> try len %d, score %d" % (len(vlist), score))
#   print("vlist %s" % (vlist))
    return score, vlist

def run_opt():
    '''Run randomized optimization with full list of independent numeric variables.
       Repeat many times to find global minimum.'''

    all_numeric_vars = ['FICO.Score', 'Amount.Requested', 'Home.Type', 'Revolving.CREDIT.Balance', 'Monthly.Income', 'Open.CREDIT.Lines', 'Debt.To.Income.Ratio', 'Loan.Length', 'Loan.Purpose.Score', 'Amount.Funded.By.Investors', 'Inquiries.in.the.Last.6.Months']
    print('\nall_vars', all_numeric_vars)

    init_list = [all_numeric_vars[0], all_numeric_vars[1]]
    opt_list = list(init_list)
    opt_score = do_naive_bayes(opt_list)
    for ix in range(len(all_numeric_vars)):
        score, vlist = random_opt(all_numeric_vars, init_list)
        if score < opt_score:
            opt_list = vlist
            opt_score = score

    print(">>> opt len %d, opt_score %d" % (len(opt_list), opt_score))
    print("opt_list %s" % (opt_list))

    do_naive_bayes(opt_list, label='opt', predict_plot=True)  # plot final optimized list


# start main script
if __name__ == '__main__':
    loansData = init_data()
    plotdir = 'naive_bayes_kfold_plots/'
    make_plotdir(plotdir)
    
    gnb = GaussianNB()
    dep_variables = ['IR_TF']
    loans_target = loansData['IR_TF']
    print('loans_target head\n', loans_target[:5])
    
    naive_bayes_tests()
    
    run_opt()
    
    print("\nConclusion: The optimum number of variables to model high vs. low interest rate\n  is five, as listed in opt_list.  Adding all eleven numeric variables or other \n  combinations decreases the prediction rate.")

