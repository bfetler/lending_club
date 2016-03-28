# -*- coding: utf-8 -*-

# SVM for lending club data, high/low interest rate prediction

import os
import pandas as pd
import numpy as np
import numpy.random as rnd
from functools import reduce
import matplotlib.pyplot as plt
from sklearn import svm, cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
import scipy.stats as sst

from naive_bayes import read_data, get_var_str


def get_app():
    return 'SVM'

def get_plotdir():
    "get plot directory"
    return 'svm_predict_plots/'

def make_plotdir():
    "make plot directory on file system"
    plotdir = get_plotdir()
    if not os.access(plotdir, os.F_OK):
        os.mkdir(plotdir)
    return plotdir

def gridscore_boxplot(gslist, plotdir, app, label, xlabel):
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
    plotfile = plotdir + "gridscore_" + label
    do_boxplot(vals, labs, app, xlabel, plotfile)

def do_boxplot(vals, labs, app, xlabel, plotfile):
    '''Create boxplot of value arrays with t-tests.'''
    is_sig = do_ttests(vals)
    if is_sig:
        sig = "Significant difference between some parameters (p-value < 0.05)"
    else:
        sig = "No significant difference in any parameters (p-value > 0.05)"
    plt.clf()
    plt.boxplot(vals, labels=labs)
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
    print("ttest pvalues", pvals)
    is_sig = list(filter(lambda e: e < 0.05, pvals))
    is_sig = (len(is_sig) > 0)
    if (not is_sig):
        print("No significant difference in any parameters (p-values > 0.05).")
    else:
        print("Significant difference between some parameters (p-values < 0.05).")
    return is_sig
    
def plot_predict(plotdir, label, app, indep_variables, pred_df, theo=False):
    '''Plot predicted correct and incorrect target values.'''
    incorrect = pred_df[ pred_df['target'] != pred_df['predict'] ]
    correct   = pred_df[ pred_df['target'] == pred_df['predict'] ]
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
    score = 100 * float(len(correct)) / pred_df.shape[0]
    txt = "Score: %.1f%% correct" % (score)
    txt2 = "%d x, %d o" % (len(incorrect), len(correct))
    plt.text(630, 42000, txt, fontsize=10)
    dbox = dict(edgecolor='black', fill=False)
    plt.text(720, 42000, txt2, bbox=dbox, fontsize=10)
    plt.text(781, 42000, 'red > 12%, blue < 12%', bbox=dbox, fontsize=10)
    txt, pos = get_var_str(indep_variables)
    plt.text(630, 38000 + 1500*(2-pos), txt, fontsize=10)
    pname = plotdir+label+'_svm_intrate_'
# needs name refactoring
    if (theo):
        pname += 'theo'
    else:
        pname += 'predict'
    plt.savefig(pname+'.png')


def load_data():
    '''Load data into dataframes, set variable list.'''
    loansData, testData = read_data()
    
    dep_variables = 'IR_TF'
    loans_y = pd.Series( loansData[dep_variables] )
    test_y = pd.Series( testData[dep_variables] )
    
    all_numeric_vars = ['FICO.Score', 'Amount.Requested', 'Home.Type', 'Revolving.CREDIT.Balance', 'Monthly.Income', 'Open.CREDIT.Lines', 'Debt.To.Income.Ratio', 'Loan.Length', 'Loan.Purpose.Score', 'Amount.Funded.By.Investors', 'Inquiries.in.the.Last.6.Months']
    print('\nall_vars\n', all_numeric_vars)
    
    loans_df = pd.DataFrame( loansData[all_numeric_vars] )
    test_df = pd.DataFrame( testData[all_numeric_vars] )
    
    return loans_df, loans_y, test_df, test_y, all_numeric_vars

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

def predict_frame(test_df, test_y, pred_y):
    '''Plot predicted data.'''
    two_variables = ['FICO.Score', 'Amount.Requested']
    pred_df = pd.DataFrame( test_df[two_variables] )
    pred_df['target'] = test_y
    pred_df['predict'] = pred_y
    return pred_df

def explore_params(loans_X, loans_y, app, plotdir):
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
    gridscore_boxplot(gs.grid_scores_, plotdir, app, "C", "kernel='linear'")
    
    clf = svm.SVC(kernel='rbf', cache_size=1000)  # 4.5 s
    param_grid = [{'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}]
    gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, \
      verbose=1, n_jobs=-1, scoring='accuracy')
    gs.fit(loans_X, loans_y)  # fit all grid parameters
    print("gs grid scores\n", gs.grid_scores_)
    print("gs best score %.5f %s\n%s" % \
      (gs.best_score_, gs.best_params_, gs.best_estimator_))
    gridscore_boxplot(gs.grid_scores_, plotdir, app, "rbf_gamma", "kernel='rbf'")

    clf = svm.SVC(kernel='rbf', cache_size=1000)  # 6.5 s
    param_grid = [{'gamma': [0.01, 0.03, 0.1, 0.3, 1.0], \
      'C': [0.3, 1.0, 3.0]}]
    gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, \
      verbose=1, n_jobs=-1, scoring='accuracy')
    gs.fit(loans_X, loans_y)  # fit all grid parameters
    print("gs grid scores\n", gs.grid_scores_)
    print("gs best score %.5f %s\n%s" % \
      (gs.best_score_, gs.best_params_, gs.best_estimator_))
    gridscore_boxplot(gs.grid_scores_, plotdir, app, "rbf_gammaC", "kernel='rbf'")

    clf = svm.LinearSVC()   # 1.6 s  fast
    param_grid = [{'C': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}]
#    do_grid_search(clf, param_grid, loans_X, loans_y)
    gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, \
      verbose=1, n_jobs=-1, scoring='accuracy')
    gs.fit(loans_X, loans_y)  # fit all grid parameters
    print("gs grid scores\n", gs.grid_scores_)
    print("gs best score %.5f %s\n%s" % \
      (gs.best_score_, gs.best_params_, gs.best_estimator_))
    gridscore_boxplot(gs.grid_scores_, plotdir, app, "LinearSVC", "LinearSVC")

def cross_validate(clf, loans_X, loans_y, print_out=False):
    '''Cross-validate fit scores.'''
    scores = cross_validation.cross_val_score(clf, loans_X, loans_y, cv=10)
    score = np.mean(scores)
    score_std = np.std(scores)
    if print_out:
        print("CV scores mean %.5f +- %.5f" % (score, 2.0 * score_std))
        print("CV raw scores", scores)
    return score, score_std, scores

def get_cv_score(clf, varlist, loans_df, loans_y):
    '''Get cross-validated score from scaled data
       selected from variable list.  Used for varlist opt.'''
    loans_X, my_scaler = scale_train_data( loans_df[varlist] )
    return cross_validate(clf, loans_X, loans_y)

def random_opt(clf, varlist, init_list, loans_df, loans_y):
    '''Optimize list by randomly adding variables,
       accept if score decreases to find local minimum.'''

    vlist = list(init_list)
    score, vstd, vscores = get_cv_score(clf, vlist, loans_df, loans_y)
    offset = len(vlist)  # offset by length of initial vlist
    indices = list(range(len(varlist) - offset))
    rnd.shuffle(indices)
    for ix in indices:
        ilist = list(vlist)
        ilist.append(varlist[ix + offset])
        iscore, istd, iscores = get_cv_score(clf, ilist, loans_df, loans_y)
        if iscore > score:
            vlist = list(ilist)
            score, vstd, vscores = iscore, istd, iscores

    print(">>> try len %d, score %.4f +- %.4f" % (len(vlist), score, vstd))
    print("vlist %s" % (vlist))
    return score, vlist, vscores

def run_opt(clf, all_numeric_vars, loans_df, loans_y, app):
    '''Run randomized optimization with full list of independent numeric variables.
       Repeat many times to find global minimum.'''

    print('\nall_vars', all_numeric_vars)
    init_list = [all_numeric_vars[0], all_numeric_vars[1]]
    opt_list = list(init_list)
    opt_score, ostd, oscores = get_cv_score(clf, opt_list, loans_df, loans_y)
    opt_raw_list = []
    for ix in range(len(all_numeric_vars)):
        score, vlist, vscores = random_opt(clf, all_numeric_vars, init_list, loans_df, loans_y)
        opt_raw_list.append({'plen': len(vlist), 'pscores': vscores})
        if score > opt_score:
            opt_list = vlist
            opt_score = score

    do_boxplot(list(map(lambda e: e['pscores'], opt_raw_list)), 
        list(map(lambda e: e['plen'], opt_raw_list)), 
        app,
        "Number of random optimized column names",
        get_plotdir() + "opt_params_boxplot")
    print(">>> opt len %d, opt_score %.4f" % (len(opt_list), opt_score))
    print("opt_list %s" % (opt_list))
    return opt_score, opt_list

# main program
def main():
    '''Main program.'''
    app = get_app()
    loans_df, loans_y, test_df, test_y, all_numeric_vars = load_data()
    indep_vars = all_numeric_vars
    loans_X, my_scaler = scale_train_data(loans_df, print_out=True)
    test_X = scale_test_data(my_scaler, test_df)
    plotdir = make_plotdir()
    
    clf = svm.SVC(kernel='linear', C=1, cache_size=1000)
    do_fit(clf, loans_X, loans_y, print_out=True)
    pred_y = do_predict(clf, test_X, test_y, print_out=True)
    pred_df = predict_frame(test_df, test_y, pred_y)    
    plot_predict(plotdir, "allvar", app, indep_vars, pred_df)
    explore_params(loans_X, loans_y, app, plotdir)
    
    clf = svm.SVC(kernel='linear', C=1, cache_size=1000)
    cross_validate(clf, loans_X, loans_y, print_out=True)
    # what's the point of above 2 lines?  test sub-method on train data?
    
    # test optimization sub-method
    indep_vars = ['FICO.Score', 'Amount.Requested', 'Home.Type']
    score, sstd, sscores = get_cv_score(clf, indep_vars, loans_df, loans_y)
    print("cv score: %.5f +- %.5f for %s" % (score, sstd, indep_vars))

#   run optimization routine    
    clf = svm.SVC(kernel='linear', C=1, cache_size=1000)
    opt_score, opt_list = run_opt(clf, all_numeric_vars, loans_df, loans_y, app)
# optimums found all have the same score within std dev: 0.89 +- 0.03
# svm is therefore less influenced by parameters chosen than naive_bayes

#   plot results of optimized list
    clf = svm.SVC(kernel='linear', C=1, cache_size=1000)
    loans_X, my_scaler = scale_train_data( loans_df[opt_list] )
    test_X = scale_test_data(my_scaler, test_df[opt_list])
    cross_validate(clf, loans_X, loans_y, print_out=True)
    
    # clf should come from opt?  or just opt_list?
    clf = svm.SVC(kernel='linear', C=1, cache_size=1000)
    do_fit(clf, loans_X, loans_y, print_out=True)
    # optimum clf model from do_fit, use in do_predict
    pred_y = do_predict(clf, test_X, test_y, print_out=True)
    pred_df = predict_frame(test_df, test_y, pred_y)    
    plot_predict(plotdir, "optvar", app, opt_list, pred_df)

if __name__ == '__main__':
    main()
    
