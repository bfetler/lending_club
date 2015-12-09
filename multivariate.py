# multivariate interactions between variables

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os

plotdir = 'multivar_plots/'
if not os.access(plotdir, os.F_OK):
    os.mkdir(plotdir)

def convert_own(s):
    if s == 'OWN':
        return 'MORTGAGE'

df = pd.read_csv('data/LoanStats3c.csv', header=1, low_memory=False)
# header=1 first line contains description
df2 = df.copy()
df2 = df2[['id','int_rate','home_ownership','annual_inc']]
# df.dropna(inplace=True)  # all rows have some NA's

df2.dropna(inplace=True)
df2['int_rate'] = df2['int_rate'].apply(lambda s: float(s.rstrip('%')))
# df2['log_income'] = np.log1p(df2.annual_inc)  # np.log1p(x) is ln(1 + x)
df2['log_income'] = np.log10(df2.annual_inc)
df2['home_ownership'] = df2['home_ownership'].apply(convert_own)
print df2[:5]
print df2[-5:]
print 'df2 shape', df2.shape


est = smf.ols(formula='int_rate ~ log_income', data=df2).fit()
print 'linear:\n', est.summary()
print 'linear params:\n', est.params

plt.clf()
plt.scatter(df2.log_income, df2.int_rate, alpha=0.3)
plt.xlabel('Log10 Income')
plt.ylabel('Interest Rate')
# plt.title('Fit: No interaction between variables')

income_linspace = np.linspace(df2.log_income.min(), df2.log_income.max(), 100)
plt.plot(income_linspace, est.params['Intercept'] + est.params['log_income'] * income_linspace, 'r')
# plt.show()
plt.savefig(plotdir+'intrate_v_income.png')


est2 = smf.ols(formula='int_rate ~ log_income * home_ownership', data=df2).fit()
print 'interaction:\n', est2.summary()
print 'interaction params:\n', est2.params

# plt.clf()
# plt.scatter(df.logincome, df.mdvis, alpha=0.3)
# plt.xlabel('Log Income')
# plt.ylabel('Number of Visits')
# plt.title('Fit: Interaction between variables')

# plt.plot(income_linspace, est.params['Intercept'] + est.params['hlthp'] * 0 + est.params['logincome'] * income_linspace + est.params['logincome:hlthp'] * 0 * income_linspace, 'r')
# plt.plot(income_linspace, est.params['Intercept'] + est.params['hlthp'] * 1 + est.params['logincome'] * income_linspace + est.params['logincome:hlthp'] * 0 * income_linspace, 'g')
# # plt.show()
# plt.savefig(plotdir+'mdvis_income_interact.png')

