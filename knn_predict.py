# -*- coding: utf-8 -*-

from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
import os

from svm_predict import load_data, do_fit, do_predict, plot_predict, \
    gridscore_boxplot, scale_train_data, scale_test_data, cross_validate, run_opt

def get_app_title():
    "get app title"
    return 'K Nearest Neighbors'

def get_app_file():
    "get app file prefix"
    return 'knn_'

def get_plotdir():
    "get plot directory"
    return 'knn_plots/'

def make_plotdir():
    "make plot directory on file system"
    plotdir = get_plotdir()
    if not os.access(plotdir, os.F_OK):
        os.mkdir(plotdir)
    return plotdir

def explore_params(loans_X, loans_y, plotdir, app, appf):
    '''Explore fit parameters on training data,
       grid search of fit scores, boxplot gridsearch results.'''
    clf = KNeighborsClassifier()
    param_grid = [{'n_neighbors': list(range(5,22,2))}]
    gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, \
      verbose=1, n_jobs=-1, scoring='accuracy')
    gs.fit(loans_X, loans_y)  # fit all grid parameters
    print("gs grid scores\n", gs.grid_scores_)
    print("gs best score %.5f %s\n%s" % \
      (gs.best_score_, gs.best_params_, gs.best_estimator_))
    gridscore_boxplot(gs.grid_scores_, plotdir, app, appf, "nn_unif", "weights=uniform")
    
    clf = KNeighborsClassifier(weights='distance')
    param_grid = [{'n_neighbors': list(range(5,22,2))}]
    gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, \
      verbose=1, n_jobs=-1, scoring='accuracy')
    gs.fit(loans_X, loans_y)  # fit all grid parameters
    print("gs grid scores\n", gs.grid_scores_)
    print("gs best score %.5f %s\n%s" % \
      (gs.best_score_, gs.best_params_, gs.best_estimator_))
    gridscore_boxplot(gs.grid_scores_, plotdir, app, appf, "nn_dist", "weights=distance")


def main():
    "main program"
    app = get_app_title()
    appf = get_app_file()
    plotdir = make_plotdir()
    
    loans_df, loans_y, test_df, test_y, numeric_vars = load_data()
    indep_vars = numeric_vars
    
    # skip scaling for now, fit score 0.68, predict score 0.64
    loans_X = loans_df
    test_X = test_df
    clf = KNeighborsClassifier(n_neighbors=11)
    do_fit(clf, loans_X, loans_y, print_out=True)
    pred_y = do_predict(clf, test_X, test_y, print_out=True)
#    plot_predict(plotdir, app, appf, "rawvar", indep_vars, test_df, test_y, pred_y)
    
    # add scaling
    loans_X, my_scaler = scale_train_data(loans_df, print_out=True)
    test_X = scale_test_data(my_scaler, test_df)
    
    # fit score 0.89, predict score 0.87
    clf = KNeighborsClassifier(n_neighbors=11)
# other params? n_neighbors, leaf_size, algorithm
    do_fit(clf, loans_X, loans_y, print_out=True)
    pred_y = do_predict(clf, test_X, test_y, print_out=True)
    plot_predict(plotdir, app, appf, "allvar", indep_vars, test_df, test_y, pred_y)
    
    # fit score 1.00, predict score 0.87, overfit?
    clf = KNeighborsClassifier(n_neighbors=11, weights='distance')
    do_fit(clf, loans_X, loans_y, print_out=True)
    pred_y = do_predict(clf, test_X, test_y, print_out=True)
    
    explore_params(loans_X, loans_y, plotdir, app, appf)
    
    clf = KNeighborsClassifier(n_neighbors=11)
    cross_validate(clf, loans_X, loans_y, print_out=True)
    
    clf = KNeighborsClassifier(n_neighbors=11)
    opt_score, opt_list = run_opt(clf, numeric_vars, loans_df, loans_y, app, appf, plotdir)
    
    loans_X, my_scaler = scale_train_data( loans_df[opt_list] )
    test_X = scale_test_data(my_scaler, test_df[opt_list])
    
    clf = KNeighborsClassifier(n_neighbors=11)
    cross_validate(clf, loans_X, loans_y, print_out=True)
    do_fit(clf, loans_X, loans_y, print_out=True)
    pred_y = do_predict(clf, test_X, test_y, print_out=True)
    plot_predict(plotdir, app, appf, "optvar", opt_list, test_df, test_y, pred_y)


if __name__ == '__main__':
    main()


