# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression as lr
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from svm_predict import load_data, do_fit, do_predict, plot_predict, \
    gridscore_boxplot, cross_validate, run_opt \
    , scale_train_data, scale_test_data

def get_app_title():
    "get app title"
    return 'Logistic Regression'

def get_app_file():
    "get app file prefix"
    return 'lr_'

def get_plotdir():
    "get plot directory"
    return 'logistic_regression_plots/'

def make_plotdir():
    "make plot directory on file system"
    plotdir = get_plotdir()
    if not os.access(plotdir, os.F_OK):
        os.mkdir(plotdir)
    return plotdir

def print_coefs(clf):
    "print logistic regression coefficients"
    print("coefs:", clf.coef_, "intercept", clf.intercept_, "n_iter", clf.n_iter_)
    # traditional coefs in "frequentist" style?

def print_lr_coefs(clf, X_labels):
    "print logistic regression coefficients"
    plist = ((lab, val) for lab, val in zip(X_labels, clf.coef_[0]))
    plist = sorted(plist, key=lambda e: np.abs(e[1]), reverse=True)
    pser = pd.Series(data = (e[1] for e in plist), index = (e[0] for e in plist))
#    print('p(x) = 1 / (1 + exp(a1*x1 + a2*x2 + b))  "logistic function"')
    print("Columns by logistic fit importance (order depends on random split)\n%s" % pser)
    print("Intercept:", clf.intercept_[0])
    return plist

def explore_params(loans_X, loans_y, plotdir, app, appf):
    '''Explore fit parameters on training data,
       grid search of fit scores, boxplot gridsearch results.'''
    clf = lr()
    param_grid = [{'C': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}]
    gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, \
      verbose=1, n_jobs=-1, scoring='accuracy')
    gs.fit(loans_X, loans_y)  # fit all grid parameters
    print("gs grid scores\n", gs.grid_scores_)
    print("gs best score %.5f %s\n%s" % \
      (gs.best_score_, gs.best_params_, gs.best_estimator_))
    gridscore_boxplot(gs.grid_scores_, plotdir, app, appf, "C", "solver='liblinear'")

def do_roc(clf, test_X, test_y, label, params, app, appf, plotdir):
    probas = clf.predict_proba(test_X)
    print('\nprobas head', probas[:3])
    fpr, tpr, thresh = roc_curve(test_y, probas[:, 1])
    print('fpr, tpr head', fpr[:3], tpr[:3])
    roc_auc = auc(fpr, tpr)
    print('roc_auc', roc_auc)
    plt.clf()
    plt.plot(fpr, tpr, lw=1, label='ROC area %.3f' % roc_auc)
    plt.text(0.6, 0.2, 'ROC area %.3f' % roc_auc)
    plt.title("High / Low Loan Rate ROC Curve Fit by " + app)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plotfile = plotdir + appf + "roc_" + label
    plt.savefig(plotfile)

def main():
    "main program"
    app = get_app_title()
    appf = get_app_file()
    plotdir = make_plotdir()
    
    loans_df, loans_y, test_df, test_y, numeric_vars = load_data()
    indep_vars = numeric_vars
    
    # skip scaling for now, score 0.71
    loans_X = loans_df
    test_X = test_df
    clf = lr()
    do_fit(clf, loans_X, loans_y, print_out=True)
    pred_y = do_predict(clf, test_X, test_y, print_out=True)  
    plot_predict(plotdir, app, appf, "rawvar", indep_vars, test_df, test_y, pred_y)

    # add scaling, score 0.90    
    loans_X, my_scaler = scale_train_data(loans_df, print_out=True)
    test_X = scale_test_data(my_scaler, test_df)
    
    clf = lr()
    do_fit(clf, loans_X, loans_y, print_out=True)
    pred_y = do_predict(clf, test_X, test_y, print_out=True)  
    plot_predict(plotdir, app, appf, "allvar", indep_vars, test_df, test_y, pred_y)
    print("columns:", indep_vars)
#   print_coefs(clf)
    X_labels = list(loans_df.columns)
#   print_lr_coefs(clf, X_labels)
    plist = print_lr_coefs(clf, indep_vars)

# find score using only top6
    top6 = [p[0] for p in plist[:6]]
    print("top6:", top6)
    loans_X = loans_df[top6]
    test_X = test_df[top6]
    loans_X, my_scaler = scale_train_data(loans_X, print_out=True)
    test_X = scale_test_data(my_scaler, test_X)
    clf = lr()
    do_fit(clf, loans_X, loans_y, print_out=True)
    pred_y = do_predict(clf, test_X, test_y, print_out=True)
    print_lr_coefs(clf, top6)
    plot_predict(plotdir, app, appf, "top6", top6, test_df, test_y, pred_y)

    do_roc(clf, test_X, test_y, "top6", top6, app, appf, plotdir)
    
#    arr = clf.decision_function(loans_df)
#    print("decision function:", arr.shape, arr)  # shape (1873,)
##    clf.decision_function(loans_df)
#    print_coefs(clf)
# traditional coefs in "frequentist" style?
#    proba = clf.predict_proba(loans_X)
#    print("proba", proba.shape, proba)
    
    explore_params(loans_X, loans_y, plotdir, app, appf)
    
    # run optimization routine
    clf = lr()
#    init_list = [indep_vars[0], indep_vars[1]]
#    random_opt(clf, indep_vars, init_list, loans_df, loans_y, print_out=True)
    opt_score, opt_list = run_opt(clf, numeric_vars, loans_df, loans_y, app, appf, plotdir, rescale=True)
    # accuracy 73% +- 3% with no scaling  (90% with scaling)
#    print_coefs(clf)

    # redo exploration with optimized columns
    loans_X = loans_df[opt_list]
    test_X = test_df[opt_list]
    loans_X, my_scaler = scale_train_data(loans_X, print_out=True)
    test_X = scale_test_data(my_scaler, test_X)
#    print("loans_X head\n", loans_X[:3])
    explore_params(loans_X, loans_y, plotdir, app, appf+"opt_")
    # accuracy 73% due to no scaling
    
    clf = lr()
    cross_validate(clf, loans_X, loans_y, print_out=True)
    
    clf = lr()
    do_fit(clf, loans_X, loans_y, print_out=True)
    pred_y = do_predict(clf, test_X, test_y, print_out=True)
    print("opt_list columns:", opt_list)
#   print_coefs(clf)
#   print_lr_coefs(clf, X_labels)
    print_lr_coefs(clf, opt_list)
    plot_predict(plotdir, app, appf, "optvar", opt_list, test_df, test_y, pred_y)
    
#    proba = clf.predict_proba(loans_X)
#    print("proba", proba.shape, proba)


if __name__ == '__main__':
    main()

