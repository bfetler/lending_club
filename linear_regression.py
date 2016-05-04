# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as linreg
from sklearn.cross_validation import cross_val_score

from svm_predict import read_data, scale_train_data, scale_test_data

def get_plotdir():
    "get plot directory"
    return 'linear_regression_plots/'

def make_plotdir():
    "make plot directory on file system"
    plotdir = get_plotdir()
    if not os.access(plotdir, os.F_OK):
        os.mkdir(plotdir)
    return plotdir

def load_data(loansData, numeric_vars):

    testData = loansData.sample(frac=0.25)
    loansData = loansData.drop(testData.index)
    
    dep_variables = 'Interest.Rate'
    train_y = pd.Series( loansData[dep_variables] )
    test_y = pd.Series( testData[dep_variables] )
    
    train_df = pd.DataFrame( loansData[numeric_vars] )
    test_df = pd.DataFrame( testData[numeric_vars] )
    
    return train_df, train_y, test_df, test_y

def sort_coefs(cols, coefs, intercept):
    "sort fit coefficients and variables"
#   df = pd.Series(coefs, index=cols).sort_values()
    plist = ((lab, val) for lab, val in zip(cols, coefs))
    plist = sorted(plist, key=lambda e: np.abs(e[1]), reverse=True)
    plist = pd.Series(data = (e[1] for e in plist), index = (e[0] for e in plist))
    print('Training Fit:\nIntercept %22s %.6f' % ('', intercept))
    print(plist)

    return plist

def get_top_vars(plist, top=5):
    new_list = plist[:top]
    return list(new_list.index)

def cross_validate(clf, train_X, train_y, cv=5, print_out=False):
    "cross-validate fit scores"
    scores = cross_val_score(clf, train_X, train_y, cv=cv)
    score = scores.mean()
    score_std = scores.std()
    if print_out:
        print("  CV scores mean %.4f +- %.4f" % (score, 2.0*score_std))
        print("  CV raw scores", scores)
    return score, scores

def run_var_list(new_vars, loansData):
    "run fit and predict with new variable list"
    train_df, train_y, test_df, test_y = load_data(loansData, new_vars)
    train_X, my_scaler = scale_train_data(train_df)
    test_X = scale_test_data(my_scaler, test_df)
    regr = linreg()
    regr.fit(train_X, train_y)
    sort_coefs(list(train_df.columns), regr.coef_, regr.intercept_)
    cross_validate(regr, train_X, train_y, cv=10, print_out=True)
    score = regr.score(train_X, train_y)
    print('Regression fit R^2 score %.4f' % score)
    pscore = regr.score(test_X, test_y)
    print('Regression predict R^2 score %.4f' % pscore)

def get_numeric_vars():
    return ['FICO.Score', 'Log.Amount.Requested', 'Home.Type', 
            'Revolving.CREDIT.Balance', 'Log.Monthly.Income', 'Log.CREDIT.Lines', 
            'Debt.To.Income.Ratio', 'Loan.Length', 'Loan.Purpose.Score', 
            'Amount.Funded.By.Investors', 'Inquiries.in.the.Last.6.Months']

def main():
    "main program"
    
    loansData = read_data()
    numeric_vars = get_numeric_vars()
    train_df, train_y, test_df, test_y = load_data(loansData, numeric_vars)
    print("train_df head\n", train_df[:3])
    print("train_y head\n", train_y[:3])
#   plotdir = make_plotdir() 

# add scaling
    train_X, my_scaler = scale_train_data(train_df)
    test_X = scale_test_data(my_scaler, test_df)
    
    regr = linreg()
    regr.fit(train_X, train_y)
#    print('regr methods', dir(regr))
#   print('columns', list(train_df.columns), 'Intercept')
#   print('coefs', regr.coef_, regr.intercept_)
    coefs = sort_coefs(list(train_df.columns), regr.coef_, regr.intercept_)

    fitpts = regr.predict(train_X)
    cross_validate(regr, train_X, train_y, cv=10, print_out=True)
    score = regr.score(train_X, train_y)
    print('Regression fit R^2 score %.4f' % score)
    
    pred = regr.predict(test_X)
#    pscore = sum(np.array(test_y) == pred)  # need np.tol.diff
    pscore = sum(np.abs(test_y - pred)) / len(test_y)
    print('Regression predict diff average %.4f' % pscore)
#    pscore = np.sqrt(sum( (test_y - pred)*(test_y - pred) ))
    pscore = regr.score(test_X, test_y)
    print('Regression predict R^2 score %.4f' % pscore)

    # try fit with fewer top variables: 5, 4, 3, 2
    for top in range(5, 1, -1):
       new_vars = get_top_vars(coefs, top)
       print('new_vars', new_vars)
       run_var_list(new_vars, loansData)

#   scores are just as good with top 4 or 5 vars as with all numeric_vars
#   scores almost as good with top 3 vars as with all numeric_vars, statistically ok
#   scores not as good with top 2 vars as with all numeric_vars, statistically not ok


if __name__ == '__main__':
    main()

