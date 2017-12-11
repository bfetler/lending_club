# -*- coding: utf-8 -*-

from sklearn.naive_bayes import GaussianNB as gnb
import os

from utils import load_data, scale_train_data, scale_test_data, \
    do_fit, do_predict, plot_predict, cross_validate, run_opt

def get_app_title():
    "get app title"
    return 'Naive Bayes'

def get_app_file():
    "get app file prefix"
    return 'gnb_'

def get_plotdir():
    "get plot directory"
    return 'naive_bayes_plots/'

def make_plotdir():
    "make plot directory on file system"
    plotdir = get_plotdir()
    if not os.access(plotdir, os.F_OK):
        os.mkdir(plotdir)
    return plotdir

def main():
    "main program"
    
    app = get_app_title()
    appf = get_app_file()
    
    loans_df, loans_y, test_df, test_y, numeric_vars = load_data()
    indep_vars = numeric_vars
    print("numeric_vars\n", numeric_vars)
    
    plotdir = make_plotdir()
    
    loans_X = loans_df
    test_X = test_df
    clf = gnb()         # skip scaling for now, score 87%
    do_fit(clf, loans_X, loans_y, print_out=True)
    pred_y = do_predict(clf, test_X, test_y, print_out=True)  
    plot_predict(plotdir, app, appf, "allvar", indep_vars, test_df, test_y, pred_y)
    
    loans_X, my_scaler = scale_train_data(loans_df, print_out=True)
    test_X = scale_test_data(my_scaler, test_df)
    clf = gnb()     # add scaling, score 87%
    do_fit(clf, loans_X, loans_y, print_out=True)
    pred_y = do_predict(clf, test_X, test_y, print_out=True)  
    plot_predict(plotdir, app, appf, "allscale", indep_vars, test_df, test_y, pred_y)
    
    # gnb has no meta-parameters to explore, optimize
    
    loans_X = loans_df
    test_X = test_df
    clf = gnb()   # score 84% +- 4%
    cross_validate(clf, loans_X, loans_y, print_out=True)
    
    clf = gnb()    # best score 89% +- 4%
    opt_score, opt_list = run_opt(clf, numeric_vars, loans_df, loans_y, app, appf, plotdir, rescale=False)
    
    # redo with optimized columns
    loans_X = loans_df[opt_list]
    test_X = test_df[opt_list]
    clf = gnb()         # best score 89% +- 4%
    cross_validate(clf, loans_X, loans_y, print_out=True)
    
    clf = gnb()         # fit score 89%, predict score 91%
    do_fit(clf, loans_X, loans_y, print_out=True)
    pred_y = do_predict(clf, test_X, test_y, print_out=True)  
    plot_predict(plotdir, app, appf, "optvar", opt_list, test_df, test_y, pred_y)

if __name__ == '__main__':
    main()

