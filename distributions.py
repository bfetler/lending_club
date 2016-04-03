# -*- coding: utf-8 -*-

import os
import numpy.random as nprnd
import scipy.stats as sst
import matplotlib.pyplot as plt

def get_plotdir():
    "get plot directory"
    return 'dist_plots/'

def make_plotdir():
    "make plot directory on file system"
    plotdir = get_plotdir()
    if not os.access(plotdir, os.F_OK):
        os.mkdir(plotdir)
    return plotdir

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
    print("t-test p-values", pvals)
    is_sig = list(filter(lambda e: e < 0.05, pvals))
    is_sig = (len(is_sig) > 0)
    if (not is_sig):
        print("No significant difference in any parameters (p-values > 0.05).")
    else:
        print("Significant difference between some parameters (p-values < 0.05).")
    return is_sig

def do_meanplot(vals, labs, app, xlabel, plotfile):
    '''Create plot of mean values without boxplot.'''
    plt.clf()
    whiten = dict(color='white')  # whis=0.0
    plt.boxplot(vals, labels=labs, showmeans=True, sym='', showcaps=False,
        showbox=False, showfliers=False, medianprops=whiten, whiskerprops=whiten, whis=0.0)
    plt.title("High / Low Loan Rate Mean Score Fit by " + app)
    plt.xlabel(xlabel)
    plt.ylabel("Fraction Correct")
    plt.tight_layout()
    plt.savefig(plotfile)

def do_boxplot(vals, labs, title, leg_title, leg_label, plotfile):
    "create boxplot of value arrays with t-tests"
    is_sig = do_ttests(vals)
    if is_sig:
        sig = "Significant difference between parameters (p-value < 0.05)"
    else:
        sig = "No significant difference between parameters (p-value > 0.05)"
    plt.clf()
    plt.boxplot(vals, labels=labs)
    plt.title(title + " With T-Test")
    plt.legend(leg_label, title=leg_title, loc='upper left', fontsize=9)
    plt.xlabel(sig)
    plt.ylabel("Values")
    plt.savefig(plotfile)

def do_histplot(vals, title, leg_title, leg_label, plotfile, bins=10):
    "create histogram plot of value arrays"
    plt.clf()
    plt.hist(vals, bins=bins)
    plt.title(title)
    plt.legend(leg_label, title=leg_title, loc='upper left', fontsize=9)
    plt.xlabel("Histograms")
    plt.ylabel("Frequency")
    plt.savefig(plotfile)

def get_data(locs=[0,0], size=50, scale=1.0):
    "get normal data distributions"
    darr = []
    for loc in locs:
        darr.append( nprnd.normal(loc=loc, scale=scale, size=size) )
    return darr

def main():
    "main routine"
    plotdir = make_plotdir()
    
    npts = 200
    title = "Normal Distributions"
    leg_t = str(npts) + " points"
    
    locs = [0, 2]
    ndata = get_data(locs=locs, size=npts)
#    print("ndata\n %s" % ndata)
    
    labs = ['a', 'b', 'c', 'd']
    labs = labs[:len(ndata)]
    
    leg_lab = list(map(lambda x: "center=%.1f" % (x) ,locs))
    do_boxplot(ndata, labs, title, leg_t, leg_lab, plotdir + "dist_diff")    
    do_histplot(ndata, title, leg_t, leg_lab, plotdir + "hist_diff", bins=20)
    # p-value ~0.096
    
    locs = [0, 0.2]
    ndata = get_data(locs=locs, size=npts)
    leg_lab = list(map(lambda x: "center=%.1f" % (x) ,locs))
    do_boxplot(ndata, labs, title, leg_t, leg_lab, plotdir + "dist_same") 
    do_histplot(ndata, title, leg_t, leg_lab, plotdir + "hist_same", bins=20)
    # p-value ~1-e67 ~0.0

if __name__ == '__main__':
    main()
