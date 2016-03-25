# test pca decomposition

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from functools import reduce
import re
import os

def load_data():
#   loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
    loansData = pd.read_csv('data/loansData.csv')  # downloaded data if no internet
    loansData.dropna(inplace=True)
    
    pat = re.compile('(.*)-(.*)')  # ()'s return two matching fields
    
    def splitSum(s):
        t = re.findall(pat, s)[0]
        return (int(t[0]) + int(t[1])) / 2
    
    sown = list(set(loansData['Home.Ownership']))
    def convert_own_to_num(s):
        return sown.index(s)
    
    loansData['Interest.Rate'] = loansData['Interest.Rate'].apply(lambda s: float(s.rstrip('%')))
    loansData['Loan.Length'] = loansData['Loan.Length'].apply(lambda s: int(s.rstrip(' months')))
    loansData['Debt.To.Income.Ratio'] = loansData['Debt.To.Income.Ratio'].apply(lambda s: float(s.rstrip('%')))
    loansData['Home.Ownership.Score'] = loansData['Home.Ownership'].apply(convert_own_to_num)
    loansData['FICO.Average'] = loansData['FICO.Range'].apply(splitSum)
    
    print('loansData head\n', loansData[:5])
    print('\nloansData basic stats\n', loansData.describe())   # print basic stats
    numeric_keys = loansData.describe().keys()  # contains numeric keys from dataframe
    print('numeric_keys\n', numeric_keys)
    
    return loansData, numeric_keys
    
def get_plotdir():
    plotdir = 'pca_explore_plots/'
    if not os.access(plotdir, os.F_OK):
        os.mkdir(plotdir)
    return plotdir

def do_pca_fit(loansData, plotname, keys, rescale=True):
    "do pca fit and fit transform"

    # mp = keys.map( lambda k: np.matrix(loansData[k]).T )
    mp = map( lambda k: np.matrix(loansData[k]).T, keys )
    X = np.column_stack(mp)

    if (rescale):
        X = StandardScaler().fit_transform(X)

    pca = PCA()
    pout = pca.fit(X)
    
    plot_fit_transform(pca, X, pout.components_, plotname, keys)
    
    return pout

def plot_fit_transform(pca, X, comps, plotname, keys):
    "do pca fit transform of original data, check components"
    pfit = pca.fit_transform(X)   # class ndarray
    print('  pfit shape', pfit.shape)
    print(pfit[:3])
    
    # check components
    print("  PCA component abs sum", list(map(lambda e: sum(np.abs(e)), comps)))
    print("  Orig component abs sum", list(map(lambda e: sum(np.abs(e)), comps.T)))
    print("  PCA comp norm?", np.allclose( \
        list(map(lambda e: sum(e*e), comps)), \
        np.ones(shape=(len(keys)))))
    print("  Orig comp norm?", np.allclose( \
        list(map(lambda e: sum(e*e), comps.T)), \
        np.ones(shape=(len(keys)))))
    print("  keys", keys)
    
    print("  fit is near equal to dot product?", np.allclose(pfit, np.dot(X, comps.T)))
#   X columns are orig: [Amount.Requested, Interest.Rate, FICO]
#   pfit columns are new: [PCA-0, PCA-1, PCA-2]
#   so comps.T columns new, rows orig: translates between the two
#   comps columns orig Xs, rows new PCAs

    # plot transformed data
    plt.clf()
    plt.plot(pfit[:,0], pfit[:,1], 'o', color='blue', alpha=0.3)
    plt.xlabel('PCA-1')
    plt.ylabel('PCA-2')
    plt.title('Lending Club Data, PCA Axes')
    plotname += '_fit' + '.png'
    plt.savefig(plotname)
    
def plot_comps(pout, plotname, keys):
    "plot pca components in PCA-0, PCA-1 plane"
    comps = pout.components_    # ndarray
    print('  comps shape', comps.shape)
    print(comps)    # print comps[0,:] # print comps[1,:]

    plt.clf()
    compx = comps[0,:]
    compy = comps[1,:]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(compx, compy, 'o', color='blue', alpha=0.5)
    plt.plot([0.0], [0.0], '+', color='black', alpha=1.0)  # center position
    for i, txt in enumerate(keys):
        ax.annotate(txt, (compx[i], compy[i]), size='x-small')
    plt.xlim([-1.2,1.2])
    plt.ylim([-1.2,1.2])
    plt.xlabel('PCA-1')
    plt.ylabel('PCA-2')
    plt.title('Lending Club, PCA Components')
    plotname += '_comps' + '.png'
    plt.savefig(plotname)
    
def plot_var_ratio(pout, plotname, keys):
    "plot pca explained variance ratios"
    varratio = pout.explained_variance_ratio_    # ndarray
    varsum = reduce(lambda x,y: x+y, varratio)
    print('  explained_variance_ratio:', varratio, ': sum =', varsum)
    vartotal = (100 * pd.Series(varratio).cumsum()).values
#   vartotal = map(lambda x: "%.1f%%" % x, vartotal)
    vartotal = list(map(lambda x: "{:.1f}%".format(x), vartotal))  # python 3 preferred
    print('  vartotal', vartotal)

    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    hscale = len(varratio)
    plt.bar(range(hscale), varratio, color='blue', align='center')
    for i, txt in enumerate(vartotal):
        ax.annotate(txt, (i-0.2+0.2/hscale, varratio[i]+0.002), size='x-small')
    plt.xlim([-0.6, hscale-0.4])
    plt.xlabel('PCA Component Number')
    plt.ylabel('Ratio')
    plt.title('Explained Variance Ratio by Component')
    plt.text(0.7, 0.85, 'Cumulative percentage of    \nexplained variance is shown',
        bbox=dict(edgecolor='black', fill=False), 
        transform=ax.transAxes, horizontalalignment='center', verticalalignment='center')
    plotname += '_var' + '.png'
    plt.savefig(plotname)

def do_pca(loansData, filename, keys, rescale=True):
    "do pca analysis and plots for set of independent variables (keys)"    

    print('do_pca', filename, ': keys', keys)
    plotname = get_plotdir() + filename
    pout = do_pca_fit(loansData, plotname, keys, rescale=True)
    plot_comps(pout, plotname, keys)
    plot_var_ratio(pout, plotname, keys)
    print('  done: %s' % filename)
    return pout

def plot_component_matrix(pout, plotdir):
    "plot component matrix (not really a confusion matrix)"
    plt.clf()
    plt.imshow(pout.components_, interpolation='nearest', cmap=plt.cm.Blues)
    plt.xlabel("Original Components")
    plt.ylabel("PCA Components")
    plt.title("PCA Component Matrix")
    plt.savefig(plotdir+"pca_component_matrix.png")

# main program
def main():
    "Main program."
    
    loansData, numeric_keys = load_data()

# what is the minimum number of features needed in PCA model?
# with all keys, reach 89% explained variance with seven components
# add more above seven, not much change
    pout = do_pca(loansData, filename='all', keys=numeric_keys)
    do_pca(loansData, filename='seven', keys=['Amount.Requested', 'Interest.Rate', 'FICO.Average', 'Debt.To.Income.Ratio', 'Monthly.Income', 'Open.CREDIT.Lines', 'Revolving.CREDIT.Balance'])
    
    plot_component_matrix(pout, get_plotdir())

if __name__ == '__main__':
    main()

