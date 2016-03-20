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


def make_plotdir():
    plotdir = 'svm_predict_plots/'
    if not os.access(plotdir, os.F_OK):
        os.mkdir(plotdir)
    return plotdir

# doesn't seem to work if called multiple times
def do_grid_search(clf, param_grid, loans_X, loans_y):
    gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, \
      verbose=1, n_jobs=-1, scoring='accuracy')
    gs.fit(loans_X, loans_y)  # fit all grid parameters
    print("gs grid scores\n", gs.grid_scores_)
    print("gs best score %.5f %s\n%s" % \
      (gs.best_score_, gs.best_params_, gs.best_estimator_))

def gridscore_boxplot(gslist, plotdir, label, xlabel):
    '''boxplot of grid score'''
    vals = list(map(lambda e: e.cv_validation_scores, gslist))
    is_sig = do_ttests(vals)
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
    if is_sig:
        sig = "Significant difference between some parameters (p-value < 0.05)"
    else:
        sig = "No significant difference in any parameters (p-value > 0.05)"
    plt.clf()
    plt.boxplot(vals, labels=labs)
    plt.title("High / Low Loan Rate Grid Score Fit by SVM")
    plt.xlabel("Parameters: " + xpar + " (with " + xlabel + ")\n" + sig)
    plt.ylabel("Fraction Correct")
    plt.tight_layout()
    plt.savefig(plotdir + "gridscore_" + label)

def do_ttests(vals):
    '''test if scores are significantly different using t-test statistics'''
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
    
def plot_predict(plotdir, label, indep_variables, pred_df, theo=False):
    '''plot predicted correct and incorrect target values'''
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
    plt.title('SVM Predicted High / Low Interest Rate')
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
    if (theo):
        pname += 'theo'
    else:
        pname += 'predict'
    plt.savefig(pname+'.png')


def load_data():
    loansData, testData = read_data()
    
    dep_variables = 'IR_TF'
    loans_y = pd.Series( loansData[dep_variables] )
    test_y = pd.Series( testData[dep_variables] )
    
    all_numeric_vars = ['FICO.Score', 'Amount.Requested', 'Home.Type', 'Revolving.CREDIT.Balance', 'Monthly.Income', 'Open.CREDIT.Lines', 'Debt.To.Income.Ratio', 'Loan.Length', 'Loan.Purpose.Score', 'Amount.Funded.By.Investors', 'Inquiries.in.the.Last.6.Months']
    print('\nall_vars', all_numeric_vars)
    
    loans_df = pd.DataFrame( loansData[all_numeric_vars] )
    test_df = pd.DataFrame( testData[all_numeric_vars] )
    
    two_variables = ['FICO.Score', 'Amount.Requested']
    pred_df = pd.DataFrame( test_df[two_variables] )
    
#    print("\nloans_df head", loans_df.__class__, loans_df.shape, "\n", loans_df[:2])
#    print("loans_y head", loans_y.__class__, loans_y.shape, "\n", loans_y[:5])
    
    return loans_df, loans_y, test_df, test_y, pred_df, all_numeric_vars

def scale_data(loans_df, test_df, print_out=False):
    scaler = StandardScaler()
    loans_X = scaler.fit_transform(loans_df)  # nparray
    test_X = scaler.transform(test_df)        # nparray
    if print_out:
        print("loans_X mean %.5f std %.5f, test_X mean %.5f std %.5f" % \
          (np.mean(loans_X), np.std(loans_X), np.mean(test_X), np.std(test_X)))
        print("scaler mean %s\nscaler std %s" % (scaler.mean_, scaler.scale_))
    # scaler.inverse_transform(result_X)

#    print("\nloans_X head", loans_X.__class__, loans_X.shape, "\n", loans_X[:2])
#    print("test_X head", test_X.__class__, test_X.shape, "\n", test_X[:2])

    return loans_X, test_X

# hmm...  just need loans_X, not test_X for opt score
# maybe I don't need to create new dataframes?
def select_data(loans_df, test_df, varlist):
    return scale_data(pd.DataFrame( loans_df[varlist] ), \
                      pd.DataFrame( test_df[varlist] ))

def fit_predict(loans_X, loans_y, test_X, test_y):
    # fit         # verbose=10 max_iter=200
    svc = svm.SVC(kernel='linear', C=0.1, cache_size=20000)
    print("svc params", svc.get_params())
    svc.fit(loans_X, loans_y)
    print("svc fit done")
    score = svc.score(loans_X, loans_y)
    print("svm score", score)  # 0.897
    
    # predict
    pred_y = svc.predict( test_X )
    pred_correct = sum(pred_y == test_y)  # 0.909
    print("pred score %.5f (%d of %d)" % \
      (pred_correct/test_y.shape[0], pred_correct, test_y.shape[0]))
    
#    return loans_X, loans_y, test_X, test_y, pred_y
    return pred_y
    
def set_predict_plot(pred_df, test_y, pred_y, indep_vars, label, plotdir):
    # plot predicted data
    pred_df['target'] = test_y
    pred_df['predict'] = pred_y
    plot_predict(plotdir, label, indep_vars, pred_df)

def explore_params(loans_X, loans_y, plotdir):
    # explore fit parameterson training data
    # grid search of fit scores
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
    gridscore_boxplot(gs.grid_scores_, plotdir, "C", "kernel='linear'")
    
    clf = svm.SVC(kernel='rbf', cache_size=1000)  # 4.5 s
    param_grid = [{'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}]
    gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, \
      verbose=1, n_jobs=-1, scoring='accuracy')
    gs.fit(loans_X, loans_y)  # fit all grid parameters
    print("gs grid scores\n", gs.grid_scores_)
    print("gs best score %.5f %s\n%s" % \
      (gs.best_score_, gs.best_params_, gs.best_estimator_))
#    is_sig = do_ttests(gs.grid_scores_)
    gridscore_boxplot(gs.grid_scores_, plotdir, "rbf_gamma", "kernel='rbf'")

    clf = svm.SVC(kernel='rbf', cache_size=1000)  # 6.5 s
    param_grid = [{'gamma': [0.01, 0.03, 0.1, 0.3, 1.0], \
      'C': [0.3, 1.0, 3.0]}]
    gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, \
      verbose=1, n_jobs=-1, scoring='accuracy')
    gs.fit(loans_X, loans_y)  # fit all grid parameters
    print("gs grid scores\n", gs.grid_scores_)
    print("gs best score %.5f %s\n%s" % \
      (gs.best_score_, gs.best_params_, gs.best_estimator_))
#    is_sig = do_ttests(gs.grid_scores_)
    gridscore_boxplot(gs.grid_scores_, plotdir, "rbf_gammaC", "kernel='rbf'")

    clf = svm.LinearSVC()   # 1.6 s  fast
    param_grid = [{'C': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}]
#    do_grid_search(clf, param_grid, loans_X, loans_y)
    gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, \
      verbose=1, n_jobs=-1, scoring='accuracy')
    gs.fit(loans_X, loans_y)  # fit all grid parameters
    print("gs grid scores\n", gs.grid_scores_)
    print("gs best score %.5f %s\n%s" % \
      (gs.best_score_, gs.best_params_, gs.best_estimator_))
#    is_sig = do_ttests(gs.grid_scores_)
    gridscore_boxplot(gs.grid_scores_, plotdir, "LinearSVC", "LinearSVC")

def cross_validate(loans_X, loans_y, print_out=False):
    # cross-validate reasonable optimum fit scores
    clf = svm.SVC(kernel='linear', C=1.0, cache_size=1000)
    scores = cross_validation.cross_val_score(clf, loans_X, loans_y, cv=10)
    score = np.mean(scores)
    score_std = np.std(scores)
    if print_out:
        print("CV scores mean %.5f +- %.5f" % (score, 2.0 * score_std))
        print("CV raw scores", scores)
    return score, score_std

def get_cv_score(indep_vars, loans_df, test_df, loans_y):
    loans_X, test_X = select_data(loans_df, test_df, indep_vars)
# could return scores, compare t-tests
    return cross_validate(loans_X, loans_y)

def random_opt(varlist, init_list, loans_df, test_df, loans_y):
    '''Optimize list by randomly adding variables,
       accept if score decreases to find local minimum.'''

    vlist = list(init_list)
    score, sstd = get_cv_score(vlist, loans_df, test_df, loans_y)
    offset = len(vlist)  # offset by length of initial vlist
    indices = list(range(len(varlist) - offset))
    rnd.shuffle(indices)
    for ix in indices:
        ilist = list(vlist)
        ilist.append(varlist[ix + offset])
        iscore, istd = get_cv_score(ilist, loans_df, test_df, loans_y)
        if iscore > score:
            vlist = list(ilist)
            score, sstd = iscore, istd

    print(">>> try len %d, score %.4f +- %.4f" % (len(vlist), score, sstd))
    print("vlist %s" % (vlist))
    return score, vlist

def run_opt(all_numeric_vars, loans_df, test_df, loans_y):
    '''Run randomized optimization with full list of independent numeric variables.
       Repeat many times to find global minimum.'''

    print('\nall_vars', all_numeric_vars)
    init_list = [all_numeric_vars[0], all_numeric_vars[1]]
    opt_list = list(init_list)
    opt_score, ostd = get_cv_score(opt_list, loans_df, test_df, loans_y)
    for ix in range(len(all_numeric_vars)):
        score, vlist = random_opt(all_numeric_vars, init_list, loans_df, test_df, loans_y)
        if score > opt_score:
            opt_list = vlist
            opt_score = score

    print(">>> opt len %d, opt_score %.4f" % (len(opt_list), opt_score))
    print("opt_list %s" % (opt_list))
    return opt_score, opt_list

# main program
if __name__ == '__main__':
    
    loans_df, loans_y, test_df, test_y, pred_df, all_numeric_vars = load_data()
    indep_vars = all_numeric_vars
    loans_X, test_X = scale_data(loans_df, test_df, print_out=True)
    plotdir = make_plotdir()
    pred_y = fit_predict(loans_X, loans_y, test_X, test_y)
    set_predict_plot(pred_df, test_y, pred_y, indep_vars, "allvar", plotdir)    
    explore_params(loans_X, loans_y, plotdir)
    cross_validate(loans_X, loans_y, print_out=True)
    
    # test opt function
    indep_vars = ['FICO.Score', 'Amount.Requested', 'Home.Type']
    score, sstd = get_cv_score(indep_vars, loans_df, test_df, loans_y)
    print("cv score: %.5f +- %.5f for %s" % (score, sstd, indep_vars))

#   run optimization routine    
    opt_score, opt_list = run_opt(all_numeric_vars, loans_df, test_df, loans_y)
# optimums found all have the same score within std dev: 0.89 +- 0.03
# svm is therefore less influenced by parameters chosen than naive_bayes

#   plot results of optimized list
    loans_X, test_X = select_data(loans_df, test_df, opt_list)
    cross_validate(loans_X, loans_y, print_out=True)
    pred_y = fit_predict(loans_X, loans_y, test_X, test_y)
    set_predict_plot(pred_df, test_y, pred_y, opt_list, "optvar", plotdir)    

