# naive bayes, classes based on logistic regression

import pandas as pd
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from functools import reduce
import re
import os

def read_data():
    "read and clean data"
    
    # loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
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
    
    loansData['Interest.Rate'] = loansData['Interest.Rate'].apply(lambda s: float(s.rstrip('%')))
    loansData['Debt.To.Income.Ratio'] = loansData['Debt.To.Income.Ratio'].apply(lambda s: float(s.rstrip('%')))
    loansData['IR_TF'] = loansData['Interest.Rate'].apply(lambda x: 0 if x<12 else 1)
    loansData['Loan.Length'] = loansData['Loan.Length'].apply(lambda s: int(s.rstrip(' months')))
    loansData['FICO.Score'] = loansData['FICO.Range'].apply(splitSum)
    loansData['Home.Type'] = loansData['Home.Ownership'].apply(own_to_num)
    loansData['Loan.Purpose.Score'] = loansData['Loan.Purpose'].apply(purpose_to_num)
    loansData['Intercept'] = 1
    
    dsize = loansData.shape[0] * 3 // 4
    testData = loansData[dsize:]
    loansData = loansData[:dsize]
    
    print('loansData head', loansData.shape, testData.shape, '\n', loansData[:5])
    print('loansData describe\n', loansData.describe())
    
    return loansData, testData

def get_plotdir():
    "get plot directory"
    return 'naive_bayes_plots/'

def make_plotdir():
    "make plot directory on file system"
    plotdir = get_plotdir()
    if not os.access(plotdir, os.F_OK):
        os.mkdir(plotdir)
    return plotdir

def plot_init(loansData):
    "plot initial data with color of interest rate class"
    plt.clf()
    plt.scatter(loansData['FICO.Score'], loansData['Amount.Requested'], c=loansData['IR_TF'], linewidths=0)
    plt.xlim(620, 850)
    plt.ylim(0, 40000)
    locs, labels = plt.yticks()
    plt.yticks(locs, map(lambda x: '$'+str(int(x/1000))+'k', locs))
    plt.xlabel('FICO Score')
    plt.ylabel('Loan Amount Requested, USD')
    plt.title('Target Interest Rates: red > 12%, blue < 12%')
    plt.savefig(get_plotdir() + 'intrate_target.png')

def get_var_str(indep_vars):
    "get variable string for plots"
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
    
def plot_predict(label, score, indep_variables, correct, incorrect, theo=False):
    '''plot predicted correct and incorrect target values'''
    plt.clf()
    plt.scatter(correct['FICO.Score'], correct['Amount.Requested'], c=correct['target'], \
         linewidths=0)
    ctag = incorrect['target']
    if (theo):
        ctag = incorrect['predict']
    plt.scatter(incorrect['FICO.Score'], incorrect['Amount.Requested'], c=ctag, \
         linewidths=1, s=20, marker='x')
    plt.xlim(620, 850)
    plt.ylim(0, 45000)
    locs, labels = plt.yticks()
    plt.yticks(locs, map(lambda x: '$'+str(int(x/1000))+'k', locs))
    plt.xlabel('FICO Score')
    plt.ylabel('Loan Amount Requested, USD')
    plt.title('Naive Bayes Predicted Interest Rate Class')
    total_pts = correct.shape[0] + incorrect.shape[0]
    sc = 100 * float(score) / total_pts
    txt = "Score: %.1f%% correct   (%d x pts)" % (sc, total_pts - score)
    plt.text(630, 42000, txt)
    plt.text(770, 42000, 'red > 12%, blue < 12%', bbox=dict(edgecolor='black', fill=False))
    txt, pos = get_var_str(indep_variables)
    plt.text(630, 38000 + 1500*(2-pos), txt, fontsize=10)
    pname = get_plotdir() + label + '_bayes_simple_intrate_'
    if (theo):
        pname += 'theo'
    else:
        pname += 'predict'
    plt.savefig(pname+'.png')

def do_naive_bayes(loansData, testData, indep_variables, label, predict_plot=False, theo_plot=False):
    "fit, predict and plot naive bayes for list of independent variables"
    print('label:', label)
    
    dep_variables = ['IR_TF']
    print('Dependent Variable(s):', dep_variables)
    print('Independent Variables:', indep_variables)
    
    loans_target = loansData['IR_TF']
    test_target = testData['IR_TF']

    loans_data = pd.DataFrame( loansData[indep_variables] )
    test_data = pd.DataFrame( testData[indep_variables] )
#    print('loans_data head\n', loans_data[:5])
#    print('test_data head\n', test_data[:5])
    
    gnb = GaussianNB()

    pred = gnb.fit(loans_data, loans_target).predict(loans_data)
    score = (loans_target == pred).sum()
    print(">>> Train: score %.1f%% correctly predicted (%d of %d points)" \
          % ( 100*score/loans_data.shape[0], score, loans_data.shape[0] ))
    print("  Number of mislabeled points : %d" % \
        (loans_target != pred).sum())
    
#    pred = gnb.fit(loans_data, loans_target).predict(test_data)
    pred = gnb.predict(test_data)
    score = (test_target == pred).sum()
    print(">>> Test: score %.1f%% correctly predicted (%d of %d points)" \
          % ( 100*score/test_data.shape[0], score, test_data.shape[0] ))
    print("    Number of mislabeled points : %d" % \
        (test_target != pred).sum())

#   data for plots
    test_data['target'] = test_target
    test_data['predict'] = pred
    incorrect = test_data[ test_data['target'] != test_data['predict'] ]
    correct = test_data[ test_data['target'] == test_data['predict'] ]
#    print("incorrect, correct shape", incorrect.shape, correct.shape)

    if (predict_plot):
        plot_predict(label, score, indep_variables, correct, incorrect)

    if (theo_plot):
        plot_predict(label, score, indep_variables, correct, incorrect, theo=True)

    return score

def main():
    "main program"
    loansData, testData = read_data()
    
    make_plotdir()
    plot_init(loansData)
    
    indep_variables = ['FICO.Score', 'Amount.Requested']
    do_naive_bayes(loansData, testData, indep_variables, label='fa', predict_plot=True, theo_plot=True)
    
    indep_variables = ['FICO.Score', 'Amount.Requested', 'Home.Type']
    do_naive_bayes(loansData, testData, indep_variables, label='fah', predict_plot=True)
    
    indep_variables = ['FICO.Score', 'Amount.Requested', 'Home.Type', 'Revolving.CREDIT.Balance', 'Monthly.Income', 'Open.CREDIT.Lines', 'Debt.To.Income.Ratio']
    do_naive_bayes(loansData, testData, indep_variables, label='all7', predict_plot=True)
    
    indep_variables = ['FICO.Score', 'Amount.Requested', 'Home.Type', 'Revolving.CREDIT.Balance', 'Monthly.Income', 'Open.CREDIT.Lines', 'Debt.To.Income.Ratio', 'Loan.Length', 'Loan.Purpose.Score', 'Amount.Funded.By.Investors', 'Inquiries.in.the.Last.6.Months']
    do_naive_bayes(loansData, testData, indep_variables, label='all', predict_plot=True)
    
    indep_variables = ['FICO.Score', 'Amount.Requested', 'Home.Type', 'Loan.Length', 'Loan.Purpose.Score', 'Amount.Funded.By.Investors', 'Inquiries.in.the.Last.6.Months']
    do_naive_bayes(loansData, testData, indep_variables, label='better', predict_plot=True)
    
    print('\nplots created')

if __name__ == '__main__':
    main()

