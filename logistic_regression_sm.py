# unit 2.4 logistic regression

import numpy as np
import numpy.random as rnd
import pandas as pd
from statsmodels.api import Logit as smLogit
from sklearn.cross_validation import KFold
#from sklearn.cross_validation import train_test_split as ttsplit
import matplotlib.pyplot as plt
import re
import os

from svm_predict import plot_predict, do_boxplot

def get_app_title():
    "get app title"
    return 'Logit Regression'

def get_app_file():
    "get app file prefix"
    return 'lr_sm_'

def make_plotdir():
    plotdir = 'logistic_sm_plots/'
    if not os.access(plotdir, os.F_OK):
        os.mkdir(plotdir)
    return plotdir

def read_data():
    # loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
    loansData = pd.read_csv('data/loansData.csv')  # downloaded data
    loansData.dropna(inplace=True)
    
    pat = re.compile('(.*)-(.*)')  # ()'s return two matching fields
    
    def splitSum(s):
    #   t = s.split('-')
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
    # loansData['FICO.Score'] = loansData['FICO.Range'].map(lambda s: int(s.split('-')[0]))
    loansData['FICO.Score'] = loansData['FICO.Range'].apply(splitSum)
    loansData['Home.Type'] = loansData['Home.Ownership'].apply(own_to_num)
    loansData['Loan.Purpose.Score'] = loansData['Loan.Purpose'].apply(purpose_to_num)
    loansData['Intercept'] = 1      # extra column needed for Logit
    
    loglist = ['Amount.Requested', 'FICO.Score', 'Monthly.Income']
    for item in loglist:
        loansData['Log.'+item] = loansData[item].apply(lambda x: np.log10(x))
    loansData['Log.CREDIT.Lines'] = loansData['Open.CREDIT.Lines'].apply(lambda x: np.log10(x))
    
    print('loansData head\n', loansData[:3])
    # print '\nloansData basic stats\n', loansData.describe()   # print basic stats
    
    # move to test()
    # print test of IR_TF calculation
    # print('loansData IntRate < 10\n', loansData[loansData['Interest.Rate'] < 10][:3])
    # print('loansData IntRate > 13\n', loansData[loansData['Interest.Rate'] > 13][:3])
    # print('loansData IntRate > 13\n', loansData[loansData['Interest.Rate'] > 13][:3])
    print('loansData IntRate < 10\n', loansData[loansData['Interest.Rate'] < 10][:3])
    print('loansData FICO > 820\n', loansData[loansData['FICO.Score'] > 820][:3])
    print('loansData FICO < 650\n', loansData[loansData['FICO.Score'] < 650][:3])
    
    return loansData

def load_data():
    loansData = read_data()
    
#    dsize = loansData.shape[0] * 3 // 4
#    testData = loansData[dsize:]
#    loansData = loansData[:dsize]

#    cols = loansData.columns
#    loansData, testData = ttsplit(loansData, test_size=0.25)
#    # return ndarrays
#    loansData = pd.DataFrame(data=loansData, columns=cols)
#    testData  = pd.DataFrame(data=testData, columns=cols)
    
    # randomization seems to wobble a bit
    testData = loansData.sample(frac=0.25)
    loansData = loansData.drop(testData.index)
    
    return loansData, testData
# don't need to split dep, indep vars here
#    return loans_df, loans_y, test_df, test_y, numeric_vars


# logistic regression example
# Q: What is the probability of getting a loan from the Lending Club
#    for $10,000 at an interest rate <= 12% with a FICO score of 750?
# Model: log(p/1-p) = mx + b  "logit function"
#   invert to find p(x) = 1 / (1 + exp(mx + b))  "logistic function"
# Wiki: The logit function is the quantile function of the
#       logistic distribution, while the probit is the quantile function
#       of the normal distribution.  See https://en.wikipedia.org/wiki/Logit

def do_logit(loansData, indep_variables, print_out=False):
    "do logit fit"
    dep_variables = ['IR_TF']
    logit = smLogit( loansData['IR_TF'], loansData[indep_variables] )
    result = logit.fit(disp=False)    # remove noisy output

    if print_out:
        print('''interest_rate = b + a1 * FICO.Score + a2 * Amount.Requested
                      = b + a1 * 750 + a2 * 10000''')
        print('find p(x) = 1 / (1 + exp(a1*x1 + a2*x2 + b))  "logistic function"')
        print('Dependent Variable(s):', dep_variables)
        print('Independent Variables:', indep_variables)
        print('fit coefficients class', result.params.__class__ , '\n', result.params)
        print('result index', result.params.index, '\nresult values', result.params.values)

    return result

def fit_score_logit(loansData, numeric_vars, cutoff=0.5):
    "fit logit and get score"
    result = do_logit(loansData, numeric_vars)
    score, loansData = calc_score(loansData, result.params, cutoff)
    return score, result, loansData

def logistic_fn_orig(loanAmount, fico, params):
    a1 = params['FICO.Score']
    a2 = params['Amount.Requested']
    b  = params['Intercept']
    p  = 1 / (1 + np.exp( b + a1 * fico + a2 * loanAmount ))
    return p

# Why do coefficients have opposite sign?  smLogit or IR_TF lambda is backwards?

def logistic_fn(loanAmount, fico, params):
    a1 = -params['FICO.Score']
    a2 = -params['Amount.Requested']
    b  = -params['Intercept']
    p  = 1 / (1 + np.exp( b + a1 * fico + a2 * loanAmount ))
    return p

# Probability isn't binary, try assumption: p<70% won't get the loan.
#   if p >= 0.70 then 1, else 0

def pred_orig(loanAmount, fico, params):
    msg = '  You will '
    p = logistic_fn(loanAmount, fico, params)
    if float(p) < 0.7:
        msg += 'NOT '
    msg += 'get the loan for under 12 percent.'
    return msg

def pred(loanAmount, fico, params):
    msg = '  You will '
    p = logistic_fn(loanAmount, fico, params)
    if float(p) > 0.3:  # IR_TF backwards?
        msg += 'NOT '
    msg += 'get the loan for under 12 percent.'
    return msg

def test_results(result):
    # could use unittest
    print('logistic values:\nloan  fico probability')
    print(10000, 750, logistic_fn(10000, 750, result.params), pred(10000, 750, result.params))
    print(10000, 720, logistic_fn(10000, 720, result.params), pred(10000, 720, result.params))
    print(10000, 710, logistic_fn(10000, 710, result.params), pred(10000, 710, result.params))
    print(10000, 700, logistic_fn(10000, 700, result.params), pred(10000, 700, result.params))
    print(10000, 690, logistic_fn(10000, 690, result.params), pred(10000, 690, result.params))
    
    print('\nThe probability that we can obtain a loan at less than 12 percent interest for $10000 USD with a FICO score of 720 is: %.1f percent.  It is more likely than not we will get the loan for under 12 percent.' % ( 100 - 100 * logistic_fn(10000, 720, result.params) ))

def plot_fico_logit(result, plotdir):
    plt.clf()
    fico_array = range(540, 860, 10)
    fico_logit = list(map(lambda x: logistic_fn(10000, x, result.params), fico_array))
    plt.plot(fico_array, fico_logit)
    plt.xlim(550, 850)
    plt.xlabel('FICO Score')
    plt.ylabel('Probability : Interest Rate > 12%')
    plt.title('Logistic Plot')
    plt.text(590, 0.25, ' Lower FICO Score ~\nHigher Interest Rate')
    plt.savefig(plotdir+'fico_logistic.png')

def plot_loan_logit(result, plotdir):
    plt.clf()
    divvy = 20
    loan_array = list(map(lambda x: 10 ** (float(x) / divvy), range(2*divvy, 5*divvy)))
    loan_logit = list(map(lambda x: logistic_fn(x, 720, result.params), loan_array))
    plt.plot(loan_array, loan_logit)
    # plt.xscale('log')
    plt.xlim(0, 40000)
    locs, labels = plt.xticks()
    plt.xticks(locs, list(map(lambda x: '$'+str(int(x/1000))+'k', locs)))
    plt.xlabel('Loan Amount Requested, USD')
    plt.ylabel('Probability : Interest Rate > 12%')
    plt.title('Logistic Plot')
    plt.text(14000, 0.25, 'Higher Loan Amount ~ Lower FICO Score\n          ~ Higher Interest Rate')
    plt.savefig(plotdir+'loan_logistic.png')

def plot_loan_fico(loansData, result, plotdir):
    plt.clf()
    plt.scatter(loansData['FICO.Score'], loansData['Amount.Requested'], c=loansData['IR_TF'], linewidths=0)
    plt.xlim(620, 850)
    plt.ylim(0, 40000)
    locs, labels = plt.yticks()
    plt.yticks(locs, list(map(lambda x: '$'+str(int(x/1000))+'k', locs)))
    plt.xlabel('FICO Score')
    plt.ylabel('Loan Amount Requested, USD')
    plt.title('Interest Rates: red > 12%, blue < 12%')
    # plt.legend(['red > 12% interest, blue < 12% interest'])
    plt.savefig(plotdir+'loan_v_fico.png')

def set_plot_predict(plotdir, app, appf, label, indep_vars, full_df):
    "set up and plot predict data"
    req = 'Amount.Requested'
    if not req in indep_vars:
        indep_vars.append(req)
    test_df = full_df[indep_vars]
    test_y  = full_df['IR_TF']
    pred_y  = full_df['Pred']
    plot_predict(plotdir, app, appf, label, indep_vars, test_df, test_y, pred_y)

def check_cutoff(dframe, vlist):
    "check optimum probability cutoff value"
    opt_score = 0
    opt_cutoff = 0.1
    cutoffs = [0.3, 0.4, 0.46, 0.48, 0.5, 0.52, 0.54, 0.6, 0.7]
    for cutoff in cutoffs:
        tr_score, result, dframe = fit_score_logit(dframe, vlist, cutoff)
        print("cutoff %.2f, score %.5f" % (cutoff, tr_score))
        if (tr_score > opt_score):
            opt_score = tr_score
            opt_cutoff = cutoff
    
    print(">>> opt cutoff %.2f, score %.5f, vars=%s" % (opt_cutoff, opt_score, vlist))

def calc_score(dframe, params, cutoff=0.5):
    '''predict values and score from fit params. 
       may be applied to train or test data.'''
    dframe['Prob'] = 0
    for par in params.index:
        dframe['Prob'] -= params[par] * dframe[par]
#    dframe['Prob'] = 1 / (1 + np.exp( dframe['Prob'] ))
    dframe['Prob'] = dframe['Prob'].apply(lambda x: 1 / (1 + np.exp(x)))
    dframe['Pred'] = dframe['Prob'].apply(lambda x: 0 if x<cutoff else 1)
    score = sum(dframe['IR_TF'] == dframe['Pred']) / dframe.shape[0]
    return score, dframe

def fit_train_score(vlist, dframe):
    "fit training data, get train score"
    tr_score, result, dframe = fit_score_logit(dframe, vlist, cutoff=0.5)
    return tr_score, result

def random_opt(varlist, init_list, dframe, print_out=False):
    '''Optimize list by randomly adding variables,
       accept if score decreases to find local minimum.'''

    vlist = list(init_list)
#    score, result = fit_train_score(vlist, dframe)
    result, scores = do_kfold_cv(dframe, vlist, n_folds=10)
    score = np.mean(scores)
    if print_out:
        print("  >>> iter init len %d, iter_score %.4f" % (len(vlist), score))
    offset = len(vlist)  # offset by length of initial vlist
    indices = list(range(len(varlist) - offset))
    rnd.shuffle(indices)
    for ix in indices:
        ilist = list(vlist)
        ilist.append(varlist[ix + offset])
#        iscore, iresult = fit_train_score(ilist, dframe)
        iresult, iscores = do_kfold_cv(dframe, ilist, n_folds=10)
        iscore = np.mean(iscores)
        if print_out:
            print("  >>> iter len %d, iter_score %.4f" % (len(ilist), iscore))
        if iscore > score:
            vlist = list(ilist)
            result = iresult
            scores = iscores
            score = iscore

    print(">>> try len %d, score %.4f" % (len(vlist), score))
    print("vlist %s" % (vlist))
    return score, vlist, result, scores

def run_opt(numeric_vars, dframe, app, appf, plotdir):
    '''Run randomized optimization with full list of independent numeric variables.
       Repeat many times to find global minimum.'''

    print('\nall_vars', numeric_vars)
    init_list = [numeric_vars[0], numeric_vars[1], numeric_vars[2]]
    opt_list = list(init_list)
#    opt_score, opt_result = fit_train_score(opt_list, dframe)
    opt_result, opt_scores = do_kfold_cv(dframe, opt_list, n_folds=10)
    opt_score = np.mean(opt_scores)
    opt_raw_list = []
    for ix in range(len(numeric_vars)):
        score, vlist, result, vscores = random_opt(numeric_vars, init_list, dframe)
        opt_raw_list.append({'plen': len(vlist), 'pscores': vscores})
        if score > opt_score:
            opt_list = vlist
            opt_score = score
            opt_scores = vscores
            opt_result = result

    do_boxplot(list(map(lambda e: e['pscores'], opt_raw_list)), 
        list(map(lambda e: e['plen'], opt_raw_list)), 
        app,
        "Number of random optimized column names",
        plotdir + appf + "opt_params_boxplot")
    print(">>> opt len %d, opt_score %.4f" % (len(opt_list), opt_score))
    print("opt_list %s" % (opt_list))
#    print("opt params\n", opt_result.params)
    print("opt params\n", opt_result)
    return opt_score, opt_list, opt_result, opt_scores

def fit_predict(loansData, validData, indep_vars):
    "fit train data, predict validation data"
    result = do_logit(loansData, indep_vars)
    score, validData = calc_score(validData, result.params)
    return score, result, validData

def do_kfold_cv(loansData, indep_vars, n_folds=5, print_out=False):
    "kfold cross-validation on train data"
    scores = []
    allpars = []
    kf = KFold(loansData.shape[0], n_folds)
    for train, test in kf:
        trainData, validData = loansData.iloc[train], loansData.iloc[test]
        validData = pd.DataFrame( validData )
        # don't really need fit_predict, just use parts
        score, result, validData = fit_predict(trainData, validData, indep_vars)
        scores.append( score )
        allpars.append( result.params )
    
    # average parameters
    llen = 1.0 / n_folds
    newpar = pd.Series( index=allpars[0].index, data=np.zeros(len(allpars[0])) )
    for par in newpar.index:
        for param in allpars:
            newpar[par] += param[par]
        newpar[par] *= llen
    
    if print_out:
        print("kfold newpar\n", newpar)
    return newpar, scores


# main program
def main():
    "Main program."
    app = get_app_title()
    appf = get_app_file()
    
    loansData, testData = load_data()
    plotdir = make_plotdir()
    
    # test steps of fit_score_logit
    indep_vars = ['FICO.Score', 'Amount.Requested', 'Intercept']
    result = do_logit(loansData, indep_vars, print_out=True)
    test_results(result)
    
    # train score
    tr_score, loansData = calc_score(loansData, result.params)
    # test score
    score, testData = calc_score(testData, result.params)
    print("testData head\n", testData[:3])
    print("score 3 vars: train %.5f, test %.5f" % (tr_score, score))
    check_cutoff(loansData, indep_vars)     # check optimum p-cutoff value
    set_plot_predict(plotdir, app, appf, "var3", indep_vars, testData)
    
    newpar, scores = do_kfold_cv(loansData, indep_vars, print_out=True)
    score, testData = calc_score(testData, newpar)
    print("score 3 kfold vars: train %.5f +- %.5f, test %.5f" % (np.mean(scores), \
        2 * np.std(scores), score))
# use kfold_cv to get error bars on test data too?
    
    plot_fico_logit(result, plotdir)
    plot_loan_logit(result, plotdir)
    plot_loan_fico(loansData, result, plotdir)
    
#   test steps of fit logit with Log.Amount.Requested
    indep_vars = ['FICO.Score', 'Log.Amount.Requested', 'Intercept']
    result = do_logit(loansData, indep_vars, print_out=True)
    
    # train score
    tr_score, loansData = calc_score(loansData, result.params)
    # test score
    score, testData = calc_score(testData, result.params)
    print("testData head\n", testData[:3])
    print("score 3 vars: train %.5f, test %.5f" % (tr_score, score))
    check_cutoff(loansData, indep_vars)     # check optimum p-cutoff value
    set_plot_predict(plotdir, app, appf, "logvar3", indep_vars, testData)
    
    newpar, scores = do_kfold_cv(loansData, indep_vars, print_out=True)
    score, testData = calc_score(testData, newpar)
    print("score 3 kfold vars: train %.5f +- %.5f, test %.5f" % (np.mean(scores), \
        2 * np.std(scores), score))

    numeric_vars = ['FICO.Score', 'Amount.Requested', 'Intercept', 'Home.Type', 'Revolving.CREDIT.Balance', 'Monthly.Income', 'Open.CREDIT.Lines', 'Debt.To.Income.Ratio', 'Loan.Length', 'Loan.Purpose.Score', 'Amount.Funded.By.Investors', 'Inquiries.in.the.Last.6.Months']
#    numeric_vars = ['FICO.Score', 'Log.Amount.Requested', 'Home.Type', 'Revolving.CREDIT.Balance', 'Log.Monthly.Income', 'Log.CREDIT.Lines', 'Debt.To.Income.Ratio', 'Loan.Length', 'Loan.Purpose.Score', 'Amount.Funded.By.Investors', 'Inquiries.in.the.Last.6.Months']
    # seems to score 85% when using LogVars, 89% when not
    tr_score, result, loansData = fit_score_logit(loansData, numeric_vars)
    score, testData = calc_score(testData, result.params)
    print("score 11 vars: train %.5f, test %.5f" % (tr_score, score))
    check_cutoff(loansData, numeric_vars)
    set_plot_predict(plotdir, app, appf, "var11", numeric_vars, testData)
    
    newpar, scores = do_kfold_cv(loansData, numeric_vars, print_out=True)
    score, testData = calc_score(testData, newpar)
    print("score 11 vars: train %.5f +- %.5f, test %.5f" % (np.mean(scores), \
        2 * np.std(scores), score))
    
    opt_score, opt_list, opt_result, opt_scores = run_opt(numeric_vars, loansData, app, appf, plotdir)
#    score, testData = calc_score(testData, opt_result.params)
    score, testData = calc_score(testData, opt_result)
    print("score opt vars: train %.5f +- %.5f, test %.5f" % (opt_score, 2 * np.std(opt_scores), score))
    check_cutoff(loansData, opt_list)
    set_plot_predict(plotdir, app, appf, "varopt", opt_list, testData)
    
    newpar, scores = do_kfold_cv(loansData, opt_list, print_out=True)
    score, testData = calc_score(testData, newpar)
    print("score opt kfold vars: train %.5f +- %.5f, test %.5f" % (np.mean(scores), \
        2 * np.std(scores), score))


if __name__ == '__main__':
    main()

