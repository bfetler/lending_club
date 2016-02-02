# test pca decomposition

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import re
import os

# loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
loansData = pd.read_csv('data/loansData.csv')  # downloaded data if no internet
loansData.dropna(inplace=True)

plotdir = 'pca_explore_plots/'
if not os.access(plotdir, os.F_OK):
    os.mkdir(plotdir)

pat = re.compile('(.*)-(.*)')  # ()'s return two matching fields

def splitSum(s):
    t = re.findall(pat, s)[0]
    return (int(t[0]) + int(t[1])) / 2

def convert_own_to_num(s):
    if s == 'RENT':
        return 0
    elif s == 'MORTGAGE':
        return 2
    elif s == 'OWN':
        return 3
    else:   # 'ANY'
        return 1

loansData['Interest.Rate'] = loansData['Interest.Rate'].apply(lambda s: float(s.rstrip('%')))
loansData['Loan.Length'] = loansData['Loan.Length'].apply(lambda s: int(s.rstrip(' months')))
loansData['Debt.To.Income.Ratio'] = loansData['Debt.To.Income.Ratio'].apply(lambda s: float(s.rstrip('%')))
loansData['Home.Ownership.Score'] = loansData['Home.Ownership'].apply(convert_own_to_num)
loansData['FICO.Average'] = loansData['FICO.Range'].apply(splitSum)

print 'loansData head\n', loansData[:5]
print '\nloansData basic stats\n', loansData.describe()   # print basic stats
numeric_keys = loansData.describe().keys()  # contains numeric keys from dataframe
print 'numeric_keys', numeric_keys

# keys = numeric_keys

pca = PCA(n_components=2)

def do_pca(filename, keys, rescale=True):

    print 'do_pca:', filename, 'keys', keys

    # mp = keys.map( lambda k: np.matrix(loansData[k]).T )
    mp = map( lambda k: np.matrix(loansData[k]).T, keys )
    X = np.column_stack(mp)

    if (rescale):
        X = StandardScaler().fit_transform(X)

#   print 'X', type(X), X.shape

    pout = pca.fit(X) # class sklearn.decomposition.pca.PCA
#   print 'n_components', pout.n_components, pout.get_params()  # boring
    comps = pout.components_  # class numpy.ndarray
    print '  comps shape', comps.shape
    print comps    # print comps[0,:] # print comps[1,:]

    # plot pca components
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
    plotname = plotdir + 'comps_' + filename + '.png'
    plt.savefig(plotname)

    pfit = pca.fit_transform(X)   # class numpy.ndarray
    print '  pfit shape', pfit.shape
    print pfit

    # plot transformed data
    plt.clf()
    plt.plot(pfit[:,0], pfit[:,1], 'o', color='blue', alpha=0.3)
    plt.xlabel('PCA-1')
    plt.ylabel('PCA-2')
    plt.title('Lending Club Data, PCA Axes')
    plotname = plotdir + 'fit_' + filename + '.png'
    plt.savefig(plotname)

    print '  plot done: %s' % filename

print ''
do_pca(filename='all', keys=numeric_keys)
do_pca(filename='three', keys=['Amount.Requested', 'Interest.Rate', 'FICO.Average'])
do_pca(filename='three_unscale', keys=['Amount.Requested', 'Interest.Rate', 'FICO.Average'], rescale=False)
do_pca(filename='nine', keys=['Amount.Requested', 'Interest.Rate', 'Loan.Length', 'Debt.To.Income.Ratio', 'Monthly.Income', 'Open.CREDIT.Lines', 'Revolving.CREDIT.Balance', 'Home.Ownership.Score', 'FICO.Average'])

# see also linear_plots/scatter_matrix.png


