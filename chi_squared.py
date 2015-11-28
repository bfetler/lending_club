# lending club data, chi-squared test

import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import collections
import os

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

loansData.dropna(inplace=True)

freq = collections.Counter(loansData['Open.CREDIT.Lines'])

plotdir = 'chisq_plots/'
if not os.access(plotdir, os.F_OK):
    os.mkdir(plotdir)

plt.clf()
plt.bar(freq.keys(), freq.values(), width=1)
plt.savefig(plotdir+'openCreditLinesBarPlot.png')

chi, p = stats.chisquare(freq.values())
print 'Open Credit Lines: chisq is %.6f, p is %.3f' % (chi, p)

