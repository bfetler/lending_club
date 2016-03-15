# -*- coding: utf-8 -*-

# SVM for lending club data, high/low interest rate prediction

import os
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from naive_bayes import read_data



def make_plotdir():
    plotdir = 'svm_predict_plots/'
    if not os.access(plotdir, os.F_OK):
        os.mkdir(plotdir)
    return plotdir

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
#    loans_X.dropna()   # does nothing
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
    
#    plotdir = make_plotdir()

    # initial test         # verbose=10 max_iter=200
    svc = svm.SVC(kernel='linear', C=0.1, cache_size=20000)
    print("svc params", svc.get_params())
    svc.fit(loans_X, loans_y)
    # Warning: using -h 0 may be faster   # need Scaler
    print("svc fit done")
    score = svc.score(loans_X, loans_y)
    print("svm score", score)
    
    pred_y = svc.predict( test_X )
    pred_correct = sum(pred_y == test_y)
    print("pred score %.5f (%d of %d)" % (pred_correct/test_y.shape[0], pred_correct, test_y.shape[0]))
    
    

