# -*- coding: utf-8 -*-

# SVM for lending club data, high/low interest rate prediction

import os
from sklearn import svm
from sklearn.grid_search import GridSearchCV
#import matplotlib.style as plt_style

from utils import load_data, scale_train_data, scale_test_data, do_fit, do_predict, \
    plot_predict, gridscore_boxplot, get_cv_score, cross_validate, run_opt

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
    
