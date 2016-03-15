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
from naive_bayes import read_data


def make_plotdir():
    plotdir = 'svm_predict_plots/'
    if not os.access(plotdir, os.F_OK):
        os.mkdir(plotdir)
    return plotdir

def gridscoreBoxplot(gslist, plotdir, label, xlabel):
    '''boxplot of grid score'''
    vals = list(map(lambda e: e.cv_validation_scores, gslist))
    labs = list(map(lambda e: list(e.parameters.values()), gslist))
    if len(labs[0]) > 1:
        labs = list(map(lambda e: reduce(lambda a,b: str(a)+"\n"+str(b), e), labs))
    else:
        # labs = list(map(lambda e: str(e[0]), labs))
        labs = [str(e[0]) for e in labs]
    xpar = list(gslist[0].parameters.keys())
    if len(xpar) > 1:
        xpar = reduce(lambda a,b: a+", "+b, xpar)
    else:
        xpar = xpar[0]
    plt.clf()
    plt.boxplot(vals, labels=labs)
    plt.title("High / Low Loan Rate Grid Score Fit by SVM")
    plt.xlabel("params: " + xpar + " (with " + xlabel + ")")
    plt.ylabel("Fraction Correct")
    plt.tight_layout()
    plt.savefig(plotdir + "gridscore_" + label)

# main program
if __name__ == '__main__':
    loansData, testData = read_data()
    
    dep_variables = ['IR_TF']
    loans_y = pd.Series( loansData['IR_TF'] )
    test_y = testData['IR_TF']
    
    all_numeric_vars = ['FICO.Score', 'Amount.Requested', 'Home.Type', 'Revolving.CREDIT.Balance', 'Monthly.Income', 'Open.CREDIT.Lines', 'Debt.To.Income.Ratio', 'Loan.Length', 'Loan.Purpose.Score', 'Amount.Funded.By.Investors', 'Inquiries.in.the.Last.6.Months']
    print('\nall_vars', all_numeric_vars)
#    indep_variables = ['FICO.Score', 'Amount.Requested']
    indep_variables = all_numeric_vars
    
    loans_X = pd.DataFrame( loansData[indep_variables] )
    test_X = pd.DataFrame( testData[indep_variables] )
    
    scaler = StandardScaler()
    loans_X = scaler.fit_transform(loans_X)
    test_X = scaler.transform(test_X)
    print("loans_X mean %.5f std %.5f, test_X mean %.5f std %.5f" % \
      (np.mean(loans_X), np.std(loans_X), np.mean(test_X), np.std(test_X)))
    print("scaler mean %s\nscaler std %s" % (scaler.mean_, scaler.scale_))
    # scaler.inverse_transform(result_X)

    print("\nloans_X head", loans_X.__class__, loans_X.shape, "\n", loans_X[:5])
    print("loans_y head", loans_y.__class__, loans_y.shape, "\n", loans_y[:5])
    
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
    
    # cross-validate initial fit scores
    clf = svm.SVC(kernel='linear', C=1.0, cache_size=1000)
    scores = cross_validation.cross_val_score(clf, loans_X, loans_y, cv=5)
    print("CV scores mean %.5f +- %.5f  (raw scores %s)" % \
      (np.mean(scores), 2.0 * np.std(scores), scores))

    # grid search of fit scores
    clf = svm.SVC(kernel='linear', cache_size=1000)  # 3.1 s
    param_grid = [{'C': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}]
    gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, \
      verbose=1, n_jobs=-1, scoring='accuracy')
    gs.fit(loans_X, loans_y)  # fit all grid parameters
    print("gs grid scores\n", gs.grid_scores_)
    print("gs best score %.5f %s\n%s" % \
      (gs.best_score_, gs.best_params_, gs.best_estimator_))
    gridscoreBoxplot(gs.grid_scores_, plotdir, "C", "kernel='linear'")
    # how to test if scores are significantly different?  stats!
    
    clf = svm.SVC(kernel='rbf', cache_size=1000)  # 4.5 s
    param_grid = [{'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}]
    gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, \
      verbose=1, n_jobs=-1, scoring='accuracy')
    gs.fit(loans_X, loans_y)  # fit all grid parameters
    print("gs grid scores\n", gs.grid_scores_)
    print("gs best score %.5f %s\n%s" % \
      (gs.best_score_, gs.best_params_, gs.best_estimator_))
    gridscoreBoxplot(gs.grid_scores_, plotdir, "rbf_gamma", "kernel='rbf'")

    clf = svm.SVC(kernel='rbf', cache_size=1000)  # 6.5 s
    param_grid = [{'gamma': [0.01, 0.03, 0.1, 0.3, 1.0], \
      'C': [0.3, 1.0, 3.0]}]
    gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, \
      verbose=1, n_jobs=-1, scoring='accuracy')
    gs.fit(loans_X, loans_y)  # fit all grid parameters
    print("gs grid scores\n", gs.grid_scores_)
    print("gs best score %.5f %s\n%s" % \
      (gs.best_score_, gs.best_params_, gs.best_estimator_))
    gridscoreBoxplot(gs.grid_scores_, plotdir, "rbf_gammaC", "kernel='rbf'")

    clf = svm.LinearSVC()   # 1.6 s  fast
    param_grid = [{'C': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}]
    gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, \
      verbose=1, n_jobs=-1, scoring='accuracy')
    gs.fit(loans_X, loans_y)  # fit all grid parameters
    print("gs grid scores\n", gs.grid_scores_)
    print("gs best score %.5f %s\n%s" % \
      (gs.best_score_, gs.best_params_, gs.best_estimator_))
    gridscoreBoxplot(gs.grid_scores_, plotdir, "LinearSVC", "LinearSVC")



