# -*- coding: utf-8 -*-

# SVM for lending club data, high/low interest rate prediction

import os
import pandas as pd
import numpy as np
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
    plt.clf()
    plt.boxplot(vals, labels=labs)
    plt.title("High / Low Loan Rate Grid Score Fit by SVM")
    plt.xlabel("parameters: " + xpar + " (with " + xlabel + ")")
    plt.ylabel("Fraction Correct")
    plt.tight_layout()
    plt.savefig(plotdir + "gridscore_" + label)

def plot_predict(label, indep_variables, pred_df, theo=False):
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

def do_ttests(gslist):
    '''test if scores are significantly different using t-test statistics'''
    vals = list(map(lambda e: e.cv_validation_scores, gslist))
#    print("grid score vals", vals)
    pvals = []
    for i, val in enumerate(vals):
        if i>0:
            pvals.append(sst.ttest_ind(vals[i-1], val).pvalue)
    qvals = [sst.ttest_ind(vals[i-1], val).pvalue if i>0 else 20 for i, val in enumerate(vals)]
    qvals.remove(20)
    print("ttest qvalues", qvals)
    print("ttest pvalues", pvals)
    isit = list(filter(lambda e: e < 0.05, pvals))
    isit = (len(isit) > 0)
    if (not isit):
        print("No significant difference in any parameters (p-values > 0.05).")
    else:
        print("Some significant differences in parameters (p-values < 0.05).")
    return isit

# main program
if __name__ == '__main__':
    loansData, testData = read_data()
    
    dep_variables = 'IR_TF'
    loans_y = pd.Series( loansData[dep_variables] )
    test_y = pd.Series( testData[dep_variables] )
    
    all_numeric_vars = ['FICO.Score', 'Amount.Requested', 'Home.Type', 'Revolving.CREDIT.Balance', 'Monthly.Income', 'Open.CREDIT.Lines', 'Debt.To.Income.Ratio', 'Loan.Length', 'Loan.Purpose.Score', 'Amount.Funded.By.Investors', 'Inquiries.in.the.Last.6.Months']
    print('\nall_vars', all_numeric_vars)
#    indep_variables = ['FICO.Score', 'Amount.Requested']
    indep_variables = all_numeric_vars
    
    loans_X = pd.DataFrame( loansData[indep_variables] )
    test_X = pd.DataFrame( testData[indep_variables] )
    
    two_variables = ['FICO.Score', 'Amount.Requested']
    pred_df = pd.DataFrame( test_X[two_variables] )
    
    print("\nloans_X head", loans_X.__class__, loans_X.shape, "\n", loans_X[:5])
    print("loans_y head", loans_y.__class__, loans_y.shape, "\n", loans_y[:5])
    print("test_X head", test_X.__class__, test_X.shape, "\n", test_X[:5])
    
    scaler = StandardScaler()
    loans_X = scaler.fit_transform(loans_X)  # nparray
    test_X = scaler.transform(test_X)        # nparray
    print("loans_X mean %.5f std %.5f, test_X mean %.5f std %.5f" % \
      (np.mean(loans_X), np.std(loans_X), np.mean(test_X), np.std(test_X)))
    print("scaler mean %s\nscaler std %s" % (scaler.mean_, scaler.scale_))
    # scaler.inverse_transform(result_X)

    print("\nloans_X head", loans_X.__class__, loans_X.shape, "\n", loans_X[:5])
    print("loans_y head", loans_y.__class__, loans_y.shape, "\n", loans_y[:5])
    print("test_X head", test_X.__class__, test_X.shape, "\n", test_X[:5])

    plotdir = make_plotdir()

    # initial fit         # verbose=10 max_iter=200
    svc = svm.SVC(kernel='linear', C=0.1, cache_size=20000)
    print("svc params", svc.get_params())
    svc.fit(loans_X, loans_y)
    print("svc fit done")
    score = svc.score(loans_X, loans_y)
    print("svm score", score)  # 0.897
    
    # initial test
    pred_y = svc.predict( test_X )
    pred_correct = sum(pred_y == test_y)  # 0.909
    print("pred score %.5f (%d of %d)" % \
      (pred_correct/test_y.shape[0], pred_correct, test_y.shape[0]))
    
    # initial plot predicted data
    pred_df['target'] = test_y
    pred_df['predict'] = pred_y
    plot_predict("allvar", indep_variables, pred_df)

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
    gridscore_boxplot(gs.grid_scores_, plotdir, "C", "kernel='linear'")

    # how to test if scores are significantly different?  stats!
    do_ttests(gs.grid_scores_)
    
    clf = svm.SVC(kernel='rbf', cache_size=1000)  # 4.5 s
    param_grid = [{'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}]
    gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, \
      verbose=1, n_jobs=-1, scoring='accuracy')
    gs.fit(loans_X, loans_y)  # fit all grid parameters
    print("gs grid scores\n", gs.grid_scores_)
    print("gs best score %.5f %s\n%s" % \
      (gs.best_score_, gs.best_params_, gs.best_estimator_))
    gridscore_boxplot(gs.grid_scores_, plotdir, "rbf_gamma", "kernel='rbf'")
    do_ttests(gs.grid_scores_)

    clf = svm.SVC(kernel='rbf', cache_size=1000)  # 6.5 s
    param_grid = [{'gamma': [0.01, 0.03, 0.1, 0.3, 1.0], \
      'C': [0.3, 1.0, 3.0]}]
    gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, \
      verbose=1, n_jobs=-1, scoring='accuracy')
    gs.fit(loans_X, loans_y)  # fit all grid parameters
    print("gs grid scores\n", gs.grid_scores_)
    print("gs best score %.5f %s\n%s" % \
      (gs.best_score_, gs.best_params_, gs.best_estimator_))
    gridscore_boxplot(gs.grid_scores_, plotdir, "rbf_gammaC", "kernel='rbf'")
    do_ttests(gs.grid_scores_)

    clf = svm.LinearSVC()   # 1.6 s  fast
    param_grid = [{'C': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}]
#    do_grid_search(clf, param_grid, loans_X, loans_y)
    gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, \
      verbose=1, n_jobs=-1, scoring='accuracy')
    gs.fit(loans_X, loans_y)  # fit all grid parameters
    print("gs grid scores\n", gs.grid_scores_)
    print("gs best score %.5f %s\n%s" % \
      (gs.best_score_, gs.best_params_, gs.best_estimator_))
    gridscore_boxplot(gs.grid_scores_, plotdir, "LinearSVC", "LinearSVC")
    do_ttests(gs.grid_scores_)
    
    # cross-validate reasonable optimum fit scores
    clf = svm.SVC(kernel='linear', C=1.0, cache_size=1000)
    scores = cross_validation.cross_val_score(clf, loans_X, loans_y, cv=10)
    print("CV scores mean %.5f +- %.5f" % (np.mean(scores), 2.0 * np.std(scores)))
    print("CV raw scores", scores)




