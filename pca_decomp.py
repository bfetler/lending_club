# test pca decomposition

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import re
import os

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
loansData.dropna(inplace=True)

plotdir = 'pca_plots/'
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
# des is a DataFrame, contains all numeric headers in loansData
# des = loansData.describe()
# keys = des.keys()
numeric_keys = loansData.describe().keys()
print 'numeric_keys', numeric_keys


# attempt pca decomposition on some variables
y  = np.matrix(loansData['Interest.Rate']).T
x1 = np.matrix(loansData['Amount.Requested']).T
x2 = np.matrix(loansData['FICO.Average']).T
x3 = np.matrix(loansData['Debt.To.Income.Ratio']).T

print 'IntRate matrix', y[:5]
print 'Amt matrix', x1[:5]
print 'FICO matrix', x2[:5]
print 'DebtRatio matrix', x3[:5]

# X = np.column_stack([x1, x2])
# X = np.column_stack([x1, x2, x3])
X = np.column_stack([x1, x2, x3, y])

# X = loansData.T  # cannot use unless all entries numeric
# X = loansData[numeric_keys]  # incorrect, way too many points in comps
# X = X.T

mp = numeric_keys.map( lambda k: np.matrix(loansData[k]).T )
print 'mp class type shape', mp.__class__, type(mp), mp.shape
X = np.column_stack(mp)



# X_std = StandardScaler().fit_transform(X)
X = StandardScaler().fit_transform(X)

print 'X', type(X), X.shape
print X[:11]
# print X[:5,:]   # same thing

pca = PCA(n_components=2)
print 'dir pca', dir(pca)
pout = pca.fit(X)
print 'pout class type', pout.__class__, type(pout)
print 'dir pout', dir(pout)
print 'pout', pout
print 'n_components', pout.n_components, pout.get_params()
comps = pout.components_
print 'comps', type(comps), len(comps), comps.size, comps.shape
print comps
print comps[0,:]
print comps[1,:]

# plot pca components
plt.clf()
compx = comps[0,:]
compy = comps[1,:]
fig = plt.figure()
ax = fig.add_subplot(111)
# plt.plot(comps[0,:], comps[1,:], 'o', color='blue', alpha=0.5)
plt.plot(compx, compy, 'o', color='blue', alpha=0.5)
plt.plot([0.0], [0.0], '+', color='black', alpha=1.0)  # center position
for i, txt in enumerate(numeric_keys):
    ax.annotate(txt, (compx[i], compy[i]), size='x-small')
plt.xlim([-1.2,1.2])
plt.ylim([-1.2,1.2])
plt.savefig(plotdir+'comps_var9y.png')

p2fit = pca.fit_transform(X)
print 'p2fit class type shape', p2fit.__class__, type(p2fit), p2fit.shape
# print 'dir p2fit', dir(p2fit)
print 'p2fit', p2fit

# plot transformed data
plt.clf()
plt.plot(p2fit[:,0], p2fit[:,1], 'o', color='blue', alpha=0.3)
plt.savefig(plotdir+'p2fit_var9y.png')

# see also linear_plots/scatter_matrix.png


