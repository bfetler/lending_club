# clean Lending Club data, find basic stats
# thinkful unit 2.2.2

import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import os

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

loansData.dropna(inplace=True)

# print loansData  # drop two rows

plotdir = 'univariate/'
if not os.access(plotdir, os.F_OK):
    os.mkdir(plotdir)

plt.clf()
loansData.boxplot(column='Amount.Funded.By.Investors')
plt.savefig(plotdir+'AmountFundedBoxplot.png')
# shows long tail up, lots of outliers

plt.clf()
loansData.hist(column='Amount.Funded.By.Investors')
plt.savefig(plotdir+'AmountFundedHistogram.png')
# shows lopsided bins, long tail up.  exponential distribution?

plt.clf()
graph = stats.probplot(loansData['Amount.Funded.By.Investors'], dist="norm", plot=plt)
plt.savefig(plotdir+'AmountFundedProbplot.png')
# QQplot shows curvature plus flat top

plt.clf()
loansData.boxplot(column='Amount.Requested')
plt.savefig(plotdir+'AmountRequestBoxplot.png')
# shows longish tail up, NO outliers

plt.clf()
loansData.hist(column='Amount.Requested')
plt.savefig(plotdir+'AmountRequestHistogram.png')
# shows lopsided bins, long tail up.  exponential distribution?

plt.clf()
graph = stats.probplot(loansData['Amount.Requested'], dist="norm", plot=plt)
plt.savefig(plotdir+'AmountRequestProbplot.png')
# QQplot shows curvature plus flat top


# try log plots
loansData['Log.Amount.Requested'] = np.log10(loansData['Amount.Requested'])
# loansData['Log.Amount.Funded'] = np.log10(loansData['Amount.Funded.By.Investors'])
# Amount.Funded may be zero, cannot take log

plt.clf()
loansData.boxplot(column='Log.Amount.Requested')
plt.savefig(plotdir+'LogAmountRequestBoxplot.png')
# shows no tail, outliers down

plt.clf()
loansData.hist(column='Log.Amount.Requested')
plt.savefig(plotdir+'LogAmountRequestHistogram.png')
# shows more or less normal distribution

plt.clf()
graph = stats.probplot(loansData['Log.Amount.Requested'], dist="norm", plot=plt)
plt.savefig(plotdir+'LogAmountRequestProbplot.png')
# QQplot shows little curvature, flat top (max 35000), flat bottom (min 1000)

# loansData['Amount.Difference'] = loansData['Amount.Requested'] - loansData['Amount.Funded.By.Investors']
# print loansData  # need to compare with other columns

