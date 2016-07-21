# -*- coding: utf-8 -*-

# SVM for lending club data, high/low interest rate prediction

import os
import re
import pandas as pd
from pandas.tools.plotting import scatter_matrix as pd_scatter_matrix
import numpy as np
import numpy.random as rnd
from functools import reduce
import matplotlib.pyplot as plt
#import matplotlib.style as plt_style
from sklearn import svm, cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
import scipy.stats as sst

def get_app_title():
    "get app title"
    return 'SVM'

def get_app_file():
    "get app file prefix"
    return 'svm_'

def get_plotdir():
    "get plot directory"
    return 'svm_predict_plots/'

def make_plotdir():
    "make plot directory on file system"
    plotdir = get_plotdir()  # add plotdir arg
    if not os.access(plotdir, os.F_OK):
        os.mkdir(plotdir)
#    plt_style.use("ggplot2")  # not found
    return plotdir

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
    
    loglist = ['Amount.Requested', 'FICO.Score', 'Monthly.Income']
    for item in loglist:
        loansData['Log.'+item] = loansData[item].apply(lambda x: np.log10(x))
    
    loansData['Log.CREDIT.Lines'] = loansData['Open.CREDIT.Lines'].apply(lambda x: np.log10(x))
# has some zeroes, -Inf
#    loansData['Log.CREDIT.Balance'] = loansData['Revolving.CREDIT.Balance'].apply(lambda x: np.log10(x))
#    loansData['Log.Amount.Funded'] = loansData['Amount.Funded.By.Investors'].apply(lambda x: np.log10(x))
    
    print('loansData head', loansData.shape, '\n', loansData[:3])
    print('\nloansData describe\n', loansData.describe())
    
    return loansData

def load_data(plotdir=""):
    '''Load data into dataframes, set variable list, split train/test data.'''
    loansData = read_data()    # add loansData arg ?
    
    numeric_vars = ['FICO.Score', 'Amount.Requested', 'Home.Type', 'Revolving.CREDIT.Balance', 'Monthly.Income', 'Open.CREDIT.Lines', 'Debt.To.Income.Ratio', 'Loan.Length', 'Loan.Purpose.Score', 'Amount.Funded.By.Investors', 'Inquiries.in.the.Last.6.Months']
    print('\nnumeric_vars\n', numeric_vars)
    
#    plt_style.use("ggplot2")  # move to make_plotdir() etc.
    loglist = ['FICO.Score', 'Amount.Requested', 'Open.CREDIT.Lines', 'Monthly.Income']
    loglist.extend(['Log.FICO.Score', 'Log.Amount.Requested', 'Log.CREDIT.Lines', 'Log.Monthly.Income'])
    
    if plotdir != "":  # stupid, but want to view entire loansData
        plot_hists(loansData, numeric_vars, "allvar", plotdir)
        plot_hists(loansData, loglist, "logvar", plotdir, ncols=4)
        plot_scatter_matrix(loansData, numeric_vars, "scatter_matrix", plotdir)
    
# log plots, last 3 look normal, Log.FICO still asymmetric.
# set new vars, still need Amount.Requested to make predict_plots
    numeric_vars = ['FICO.Score', 'Amount.Requested', 'Log.Amount.Requested', 'Home.Type', 'Revolving.CREDIT.Balance', 'Log.Monthly.Income', 'Log.CREDIT.Lines', 'Debt.To.Income.Ratio', 'Loan.Length', 'Loan.Purpose.Score', 'Amount.Funded.By.Investors', 'Inquiries.in.the.Last.6.Months']
    
#    dsize = loansData.shape[0] * 3 // 4
#    testData = loansData[dsize:]
#    loansData = loansData[:dsize]
# some differences, use train_test_split()
    
#    loansData, testData = cross_validation.train_test_split(loansData, test_size=0.25)
    # returns ndarrays

    testData = loansData.sample(frac=0.25)
    loansData = loansData.drop(testData.index)
    
    print('\nloansData describe\n', loansData.describe())
    print('\ntestData describe\n', testData.describe())
    
    dep_variables = 'IR_TF'
    loans_y = pd.Series( loansData[dep_variables] )
    test_y = pd.Series( testData[dep_variables] )
    
    loans_df = pd.DataFrame( loansData[numeric_vars] )
    test_df = pd.DataFrame( testData[numeric_vars] )
    
# now that frames contain all numeric_vars, remove one to use for fitting
    numeric_vars.remove('Amount.Requested')
    
    return loans_df, loans_y, test_df, test_y, numeric_vars

def plot_hists(df, vlist, label, plotdir, ncols=3):
    plt.clf()
    nrows = len(vlist) // ncols
    if len(vlist) % ncols > 0:
        nrows += 1
    for i, var in enumerate(vlist):
        plt.subplot(nrows, ncols, i+1)
        plt.hist(df[var], bins=30)
        plt.title(var, fontsize=10)
        plt.tick_params(labelbottom='off', labelleft='off')
    plt.savefig(plotdir + 'hist_' + label + '.png')

def plot_scatter_matrix(df, numeric_vars, label, plotdir):
    plt.clf()
    pd_scatter_matrix(df[numeric_vars])  # can't set fontsize=6
    plt.savefig(plotdir + label + '.png')

def gridscore_boxplot(gslist, plotdir, app, appf, label, xlabel):
    '''Set up boxplot of grid scores.'''
    vals = list(map(lambda e: e.cv_validation_scores, gslist))
    labs = list(map(lambda e: list(e.parameters.values()), gslist))
    if len(labs[0]) > 1:
        labs = list(map(lambda e: reduce(lambda a,b: str(a)+"\n"+str(b), e), labs))
    else:
        # labs = list(map(lambda e: str(e[0]), labs))
        labs = [str(lab[0]) for lab in labs]
    xpar = list(gslist[0].parameters.keys())
    if len(xpar) > 1:
        xpar = reduce(lambda a,b: a+", "+b, xpar)
    else:
        xpar = xpar[0]
    xlabel = "Parameters: " + xpar + " (with " + xlabel + ")"
    plotfile = plotdir + appf + "gridscore_" + label
#    plotfile = "%s%s%s%s" % (plotdir, appf, "gridscore_", label) # faster?
    # or "".join(slist)   # str is immutable
    do_boxplot(vals, labs, app, xlabel, plotfile)
#    do_meanplot(vals, labs, app, xlabel, plotfile+"_mean")

def do_boxplot(vals, labs, app, xlabel, plotfile):
    '''Create boxplot of value arrays with t-tests.'''
    is_sig = do_ttests(vals)
    if is_sig:
        sig = "Significant difference between some parameters (p-value < 0.05)"
    else:
        sig = "No significant difference in any parameters (p-value > 0.05)"
    plt.clf()
    plt.boxplot(vals, labels=labs)
#   plt.boxplot(vals, labels=labs, showmeans=True)
    # showmeans=True  useful for interp run_opt
    plt.title("High / Low Loan Rate Grid Score Fit by " + app)
    plt.xlabel(xlabel + "\n" + sig)
    plt.ylabel("Fraction Correct")
    plt.tight_layout()
    plt.savefig(plotfile)

def do_ttests(vals):
    '''Test if scores are significantly different using t-test statistics.'''
#    vals = list(map(lambda e: e.cv_validation_scores, gslist))
    pvals = []
    for i, val in enumerate(vals):
        if i>0:
            pvals.append(sst.ttest_ind(vals[i-1], val).pvalue)
#    qvals = [sst.ttest_ind(vals[i-1], val).pvalue if i>0 else 20 \
#      for i, val in enumerate(vals)]
#    qvals.remove(20)
    print("t-test p-values", pvals)
    is_sig = list(filter(lambda e: e < 0.05, pvals))
    is_sig = (len(is_sig) > 0)
    if (not is_sig):
        print("No significant difference in any parameters (p-values > 0.05).")
    else:
        print("Significant difference between some parameters (p-values < 0.05).")
    return is_sig

def do_meanplot(vals, labs, app, xlabel, plotfile):
    '''Create plot of mean values without boxplot.'''
    plt.clf()
    whiten = dict(color='white')  # whis=0.0
    plt.boxplot(vals, labels=labs, showmeans=True, sym='', showcaps=False,
        showbox=False, showfliers=False, medianprops=whiten, whiskerprops=whiten, whis=0.0)
    plt.title("High / Low Loan Rate Mean Score Fit by " + app)
    plt.xlabel(xlabel)
    plt.ylabel("Fraction Correct")
    plt.tight_layout()
    plt.savefig(plotfile)

def predict_frame(test_df, test_y, pred_y):
    '''Create correct, incorrect prediction dataframe.'''
    two_vars = ['FICO.Score', 'Amount.Requested']
    pred_df = pd.DataFrame( test_df[two_vars] )
    pred_df['target'] = test_y
    pred_df['predict'] = pred_y
    incorrect = pred_df[ pred_df['target'] != pred_df['predict'] ]
    correct   = pred_df[ pred_df['target'] == pred_df['predict'] ]
    return correct, incorrect

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

def plot_predict(plotdir, app, appf, label, indep_vars, test_df, test_y, pred_y, theo=False):
    '''Plot predicted correct and incorrect target values.'''
    correct, incorrect = predict_frame(test_df, test_y, pred_y)
    
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
    plt.title('High / Low Interest Rate Predicted by ' + app)
    score = 100 * float(len(correct)) / test_df.shape[0]
    txt = "Score: %.1f%% correct" % (score)
    txt2 = "%d x, %d o" % (len(incorrect), len(correct))
    plt.text(630, 42000, txt, fontsize=10)
    dbox = dict(edgecolor='black', fill=False)
    plt.text(720, 42000, txt2, bbox=dbox, fontsize=10)
    plt.text(781, 42000, 'red > 12%, blue < 12%', bbox=dbox, fontsize=10)
    txt, pos = get_var_str(indep_vars)
    plt.text(630, 38000 + 1500*(2-pos), txt, fontsize=10)
    pname = plotdir+ appf + 'intrate_' + label
    if (theo):
        pname += '_theo'
    else:
        pname += '_predict'
    plt.savefig(pname+'.png')


def scale_train_data(loans_df, print_out=False):
    '''Scale data for svm transform, read dataframe, return nparray.'''
    scaler = StandardScaler()
    loans_X = scaler.fit_transform(loans_df)  # nparray
    if print_out:
        print("loans_X mean %.5f std %.5f" % \
          (np.mean(loans_X), np.std(loans_X)))
        print("scaler mean %s\nscaler std %s" % (scaler.mean_, scaler.scale_))
    # scaler.inverse_transform(result_X)
    return loans_X, scaler

def scale_test_data(scaler, test_df):
    return scaler.transform(test_df)

def do_fit(clf, loans_X, loans_y, print_out=False):
    '''Fit training data.  Only for modeling use.'''
    clf.fit(loans_X, loans_y)
    fit_score = clf.score(loans_X, loans_y)
    if print_out:
        print("params", clf.get_params())
        print("fit done, score %.5f" % fit_score)  # 0.897
    return fit_score

def do_predict(clf, test_X, test_y, print_out=False):
    '''Predict test data.  For modeling and production use.'''
    pred_y = clf.predict( test_X )
    pred_correct = sum(pred_y == test_y)  # 0.909
    pred_score = pred_correct/test_y.shape[0]
    if print_out:
        print("pred score %.5f (%d of %d)" % \
          (pred_score, pred_correct, test_y.shape[0]))
    return pred_y

def explore_params(loans_X, loans_y, plotdir, app, appf):
    '''Explore fit parameters on training data,
       grid search of fit scores, boxplot gridsearch results.'''
    clf = svm.SVC(kernel='linear', cache_size=1000)  # 3.1 s
    param_grid = [{'C': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}]
    gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, \
      verbose=1, n_jobs=-1, scoring='accuracy')
    gs.fit(loans_X, loans_y)  # fit all grid parameters
    print("gs grid scores\n", gs.grid_scores_)
    print("gs best score %.5f %s\n%s" % \
      (gs.best_score_, gs.best_params_, gs.best_estimator_))
    # how to test if scores are significantly different?  stats!
#    is_sig = do_ttests(gs.grid_scores_)
    gridscore_boxplot(gs.grid_scores_, plotdir, app, appf, "C", "kernel='linear'")
    
    clf = svm.SVC(kernel='rbf', cache_size=1000)  # 4.5 s
    param_grid = [{'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}]
    gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, \
      verbose=1, n_jobs=-1, scoring='accuracy')
    gs.fit(loans_X, loans_y)  # fit all grid parameters
    print("gs grid scores\n", gs.grid_scores_)
    print("gs best score %.5f %s\n%s" % \
      (gs.best_score_, gs.best_params_, gs.best_estimator_))
    gridscore_boxplot(gs.grid_scores_, plotdir, app, appf, "rbf_gamma", "kernel='rbf'")

    clf = svm.SVC(kernel='rbf', cache_size=1000)  # 6.5 s
    param_grid = [{'gamma': [0.01, 0.03, 0.1, 0.3, 1.0], \
      'C': [0.3, 1.0, 3.0]}]
    gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, \
      verbose=1, n_jobs=-1, scoring='accuracy')
    gs.fit(loans_X, loans_y)  # fit all grid parameters
    print("gs grid scores\n", gs.grid_scores_)
    print("gs best score %.5f %s\n%s" % \
      (gs.best_score_, gs.best_params_, gs.best_estimator_))
    gridscore_boxplot(gs.grid_scores_, plotdir, app, appf, "rbf_gammaC", "kernel='rbf'")

    clf = svm.LinearSVC()   # 1.6 s  fast
    param_grid = [{'C': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}]
    gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, \
      verbose=1, n_jobs=-1, scoring='accuracy')
    gs.fit(loans_X, loans_y)  # fit all grid parameters
    print("gs grid scores\n", gs.grid_scores_)
    print("gs best score %.5f %s\n%s" % \
      (gs.best_score_, gs.best_params_, gs.best_estimator_))
    gridscore_boxplot(gs.grid_scores_, plotdir, app, appf, "LinearSVC", "LinearSVC")

def cross_validate(clf, loans_X, loans_y, print_out=False):
    '''Cross-validate fit scores.'''
    scores = cross_validation.cross_val_score(clf, loans_X, loans_y, cv=10, scoring='accuracy')
# scoring = 'accuracy' | 'f1'
    score = np.mean(scores)
    score_std = np.std(scores)
    if print_out:
        print("CV scores mean %.5f +- %.5f" % (score, 2.0 * score_std))
        print("CV raw scores", scores)
    # return dict ?
    return score, score_std, scores

def get_cv_score(clf, varlist, loans_df, loans_y, rescale=True):
    '''Get cross-validated score from scaled data
       selected from variable list.  Used for varlist opt.'''
    loans_X = loans_df[varlist]
    if rescale:
        loans_X, my_scaler = scale_train_data( loans_X )
    return cross_validate(clf, loans_X, loans_y)

def random_opt(clf, varlist, init_list, loans_df, loans_y, score_fn=get_cv_score, rescale=True, print_out=False):
    '''Optimize list by randomly adding variables,
       accept if score decreases to find local minimum.'''

    vlist = list(init_list)
    score, vstd, vscores = score_fn(clf, vlist, loans_df, loans_y, rescale)
    if print_out:
        print("  >>> iter init len %d, iter_score %.4f" % (len(vlist), score))
    offset = len(vlist)  # offset by length of initial vlist
    indices = list(range(len(varlist) - offset))
    rnd.shuffle(indices)
    for ix in indices:
        ilist = list(vlist)
        ilist.append(varlist[ix + offset])
        iscore, istd, iscores = score_fn(clf, ilist, loans_df, loans_y, rescale)
        if print_out:
            print("  >>> iter len %d, iter_score %.4f" % (len(ilist), iscore))
        if iscore > score:
            vlist = list(ilist)
            score, vstd, vscores = iscore, istd, iscores

    print(">>> try len %d, score %.4f +- %.4f" % (len(vlist), score, 2 * vstd))
    print("vlist %s" % (vlist))
    # return dict ?
    return score, vlist, vscores

def run_opt(clf, numeric_vars, loans_df, loans_y, app, appf, plotdir, score_fn=get_cv_score, rescale=True):
    '''Run randomized optimization with full list of independent numeric variables.
       Repeat many times to find global minimum.'''

    print('\nall_vars', numeric_vars)
    print(">>> run_opt clf params", clf.get_params())
    init_list = [numeric_vars[0], numeric_vars[1]]
    opt_list = list(init_list)
    opt_score, ostd, oscores = score_fn(clf, opt_list, loans_df, loans_y, rescale)
    opt_raw_list = []
    for ix in range(len(numeric_vars)):
        score, vlist, vscores = random_opt(clf, numeric_vars, init_list, loans_df, loans_y, score_fn, rescale)
        opt_raw_list.append({'plen': len(vlist), 'pscores': vscores})
        if score > opt_score:
            opt_list = vlist
            opt_score = score

    do_boxplot(list(map(lambda e: e['pscores'], opt_raw_list)), 
        list(map(lambda e: e['plen'], opt_raw_list)), 
        app,
        "Number of random optimized column names",
        plotdir + appf + "opt_params_boxplot")
#    do_meanplot(list(map(lambda e: e['pscores'], opt_raw_list)), 
#        list(map(lambda e: e['plen'], opt_raw_list)), 
#        app,
#        "Number of random optimized column names",
#        plotdir + appf + "opt_params_meanplot")
    
    print(">>> opt len %d, opt_score %.4f" % (len(opt_list), opt_score))
    print("opt_list %s" % (opt_list))
    return opt_score, opt_list

# main program
def main():
    '''Main program.'''
    app = get_app_title()
    appf = get_app_file()
    plotdir = make_plotdir()
    
    loans_df, loans_y, test_df, test_y, numeric_vars = load_data()
    indep_vars = numeric_vars
    loans_X, my_scaler = scale_train_data(loans_df, print_out=True)
    test_X = scale_test_data(my_scaler, test_df)
    
    clf = svm.SVC(kernel='linear', C=1, cache_size=1000)
    do_fit(clf, loans_X, loans_y, print_out=True)
    pred_y = do_predict(clf, test_X, test_y, print_out=True)   
    plot_predict(plotdir, app, appf, "allvar", indep_vars, test_df, test_y, pred_y)
    
    explore_params(loans_X, loans_y, plotdir, app, appf)
    
    # test optimization sub-method
    clf = svm.SVC(kernel='linear', C=1, cache_size=1000)
    indep_vars = ['FICO.Score', 'Amount.Requested', 'Home.Type']
    score, sstd, sscores = get_cv_score(clf, indep_vars, loans_df, loans_y)
    print("cv score: %.5f +- %.5f for %s" % (score, 2.0 * sstd, indep_vars))

#   run optimization routine
    clf = svm.SVC(kernel='linear', C=1, cache_size=1000)
    opt_score, opt_list = run_opt(clf, numeric_vars, loans_df, loans_y, app, appf, plotdir)
# optimums found all have the same score within std dev: 0.89 +- 0.03
# svm is therefore less influenced by parameters chosen than naive_bayes

#   repeat results of optimized list and plot
    clf = svm.SVC(kernel='linear', C=1, cache_size=1000)
    loans_X, my_scaler = scale_train_data( loans_df[opt_list] )
    test_X = scale_test_data(my_scaler, test_df[opt_list])
    cross_validate(clf, loans_X, loans_y, print_out=True)
    
    # clf should come from opt?  or just opt_list?
    clf = svm.SVC(kernel='linear', C=1, cache_size=1000)
    do_fit(clf, loans_X, loans_y, print_out=True)
    # optimum clf model from do_fit, use in do_predict
    pred_y = do_predict(clf, test_X, test_y, print_out=True)
    plot_predict(plotdir, app, appf, "optvar", opt_list, test_df, test_y, pred_y)

if __name__ == '__main__':
    main()
    
