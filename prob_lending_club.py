# clean Lending Club data, find basic stats

import pandas as pd
import scipy.stats as stats
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
plt.savefig(plotdir+'boxplotAmountFunded.png')
# shows long tail up, lots of outliers

plt.clf()
loansData.hist(column='Amount.Funded.By.Investors')
plt.savefig(plotdir+'histAmountFunded.png')
# shows lopsided bins, long tail up. exponential distribution?

plt.clf()
graph = stats.probplot(loansData['Amount.Funded.By.Investors'], dist="norm", plot=plt)
plt.savefig(plotdir+'probplotAmountFunded.png')
# QQplot shows curvature plus flat top



