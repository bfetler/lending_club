# full data pca decomposition

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import re
import os

print 'reading csv data ...',
df = pd.read_csv('data/LoanStats3b.csv', header=1, low_memory=False)
print 'copying data ...',
df2 = df.copy()
print 'copy done'
print 'df2 shape', df2.shape

# really should download it, if no internet connection
# loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
df2.dropna(axis=1, how='all', inplace=True)  # drop columns that are all NA
print 'df2 after drop1 shape', df2.shape
# df2.fillna(0, inplace=True)
# print 'df2 after fillna shape', df2.shape
# too many zeroes alters PCA fit drastically
df2.dropna(axis=0, how='any', inplace=True)  # drop rows that have an NA
print 'df2 after drop2 shape', df2.shape

plotdir = 'pca_full_plots/'
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

# fails if fillna(0) not a date
# df2['issue_d_format'] = pd.to_datetime(df2['issue_d'], format='%b-%Y')
# df2 = df2.set_index('issue_d_format')
loansData = df2

loansData['int_rate'] = loansData['int_rate'].apply(lambda s: float(str(s).rstrip('%')))
# loansData['term'] = loansData['term'].apply(lambda s: int(str(s).rstrip(' months')))
# loansData['Debt.To.Income.Ratio'] = loansData['Debt.To.Income.Ratio'].apply(lambda s: float(s.rstrip('%')))
# loansData['Home.Ownership.Score'] = loansData['Home.Ownership'].apply(convert_own_to_num)
# loansData['FICO.Average'] = loansData['FICO.Range'].apply(splitSum)

print 'loansData head\n', loansData[:5]
print '\nloansData basic stats\n', loansData.describe()   # print basic stats

# no FICO
print 'loansData keys\n', loansData.keys().shape, loansData.keys()
# des is a DataFrame, contains all numeric headers in loansData
# des = loansData.describe()
# keys = des.keys()
numeric_keys = loansData.describe().keys()
print 'numeric_keys', numeric_keys.shape, numeric_keys

# convert other non-numeric keys to numeric, e.g. term, home_ownership, 
#     emp_length, loan_status, purpose, zip_code, application_type
# pick and choose which ones to include
# check boxplot of each variable of interest

pca = PCA()

# keys = numeric_keys

def do_pca(filename, keys, rescale=True):

    print 'do_pca:', filename, 'keys', len(keys), keys

#   mp = keys.map( lambda k: np.matrix(loansData[k]).T )
    mp = map( lambda k: np.matrix(loansData[k]).T , keys )
#   print 'mp class type shape', mp.__class__, type(mp), mp.shape
    X = np.column_stack(mp)
#   mp, X both class numpy.ndarray

    # transform scales to mean=0 sd=1
    if (rescale):
        X = StandardScaler().fit_transform(X)

#   print 'X', X.__class__, type(X), X.shape
#   print X[:5]

    pout = pca.fit(X)   # class sklearn.decomposition.pca.PCA
    comps = pout.components_
    print '  ', filename, 'comps', comps   # print comps[0,:]  # print comps[1,:]
    varratio = pout.explained_variance_ratio_
    varsum = reduce(lambda x,y: x+y, varratio)
    print '  explained_variance_ratio:', varratio.__class__, varratio, ': sum =', varsum
    vartotal = (100 * pd.Series(varratio).cumsum()).values
    vartotal = map(lambda x: "{:.1f}%".format(x), vartotal)
    print '  vartotal', vartotal.__class__, vartotal

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
    plotname = plotdir + filename + '_comps' + '.png'
    plt.savefig(plotname)

    # plot pca component ratios
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
    plotname = plotdir + filename + '_var' + '.png'
    plt.savefig(plotname)

    pfit = pca.fit_transform(X)   # class numpy.ndarray
#   print 'pfit class type shape', pfit.__class__, type(pfit), pfit.shape

    # plot transformed data
    plt.clf()
    plt.plot(pfit[:,0], pfit[:,1], 'o', color='blue', alpha=0.3)
    plt.xlabel('PCA-1')
    plt.ylabel('PCA-2')
    plt.title('Lending Club Data, PCA Axes')
    plotname = plotdir + filename + '_fit' + '.png'
    plt.savefig(plotname)

    print '  plots done: %s' % filename


do_pca(filename='four', keys=['funded_amnt', 'int_rate', 'annual_inc', 'installment'])
do_pca(filename='six', keys=['funded_amnt', 'int_rate', 'annual_inc', 'installment', 'revol_bal', 'total_acc'])
do_pca(filename='all', keys=numeric_keys)

# see also linear_plots/scatter_matrix.png

