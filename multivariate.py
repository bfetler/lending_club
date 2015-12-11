# multivariate interactions between variables

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import collections
import os

plotdir = 'multivar_plots/'
if not os.access(plotdir, os.F_OK):
    os.mkdir(plotdir)

def convert_own_to_num(s):
    if s == 'RENT':
        return 0
    elif s == 'MORTGAGE':
        return 2
    elif s == 'OWN':
        return 3
    else:   # 'ANY'
#       return 1
        return np.NaN

def convert_own_to_color(s):
    if s == 'RENT':
        return 'red'
    elif s == 'MORTGAGE':
        return 'blue'
    elif s == 'OWN':
        return 'green'
    else:    # 'ANY'
#       return 'magenta'
        return np.NaN

df = pd.read_csv('data/LoanStats3c.csv', header=1, low_memory=False)
# header=1 first line contains description
df2 = df.copy()
df2 = df2[['id','int_rate','home_ownership','annual_inc']]
# df.dropna(inplace=True)  # careful, all rows have some NA's

df2.dropna(inplace=True)  # drops 4 rows
df2['int_rate'] = df2['int_rate'].apply(lambda s: float(s.rstrip('%')))
# df2['log_income'] = np.log1p(df2.annual_inc)  # np.log1p(x) is ln(1 + x)
df2['log_income'] = np.log10(df2.annual_inc)
df2['home_type'] = df2['home_ownership'].apply(convert_own_to_num)
df2['home_color'] = df2['home_ownership'].apply(convert_own_to_color)
print df2[:5]
print df2[-5:]
print 'df2 shape', df2.shape
print 'home_ownership values:', set(df2['home_ownership'].tolist())

freq = collections.Counter(df2['home_ownership'])
print 'home_ownership frequency:', freq.keys(), freq.values()
freq = collections.Counter(df2['home_type'])
# plt.bar() requires int/float, need numeric version of home_ownership
print 'home_type frequency [RENT, ANY, MORTGAGE, OWN] :', freq.keys(), freq.values()
plt.clf()
# df2.groupby('home_ownership').hist()
plt.bar(freq.keys(), freq.values(), width=1)
# add labels to bar plot
# plt.show()
# plt.xlabel('Log10 Income')
plt.title('RENT                ANY           MORTGAGE            OWN')
plt.ylabel('Frequency')
plt.savefig(plotdir+'home_barplot.png')

# log_income is normally distributed, annual_inc is not
df2.boxplot(by='home_ownership', column='annual_inc')
plt.text(1, 6500000, 'Annual income not normally distributed, try log')
plt.savefig(plotdir+'home_income_boxplot.png')
df2.boxplot(by='home_ownership', column='log_income')
plt.savefig(plotdir+'home_logincome_boxplot.png')
df2.boxplot(by='home_ownership', column='int_rate')
plt.savefig(plotdir+'home_intrate_boxplot.png')

df2.dropna(inplace=True)  # drop 1 row w/ ANY, can't do stats on it
print 'df2 shape', df2.shape
print 'home_ownership values:', set(df2['home_ownership'].tolist())

est = smf.ols(formula='int_rate ~ log_income', data=df2).fit()
print 'linear:\n', est.summary()
print 'linear params:\n', est.params

plt.clf()
# plt.scatter(df2.log_income, df2.int_rate, alpha=0.3)
plt.scatter(df2.log_income, df2.int_rate, alpha=0.3, c=df2['home_color'], linewidths=0)
plt.xlabel('Log10 Income')
plt.ylabel('% Interest Rate')
plt.title('Fit: without home_ownership')

income_linspace = np.linspace(df2.log_income.min(), df2.log_income.max(), 100)
plt.plot(income_linspace, est.params['Intercept'] + est.params['log_income'] * income_linspace, 'r')
# plt.show()
plt.savefig(plotdir+'intrate_v_income.png')


est2 = smf.ols(formula='int_rate ~ log_income * home_type', data=df2).fit()
print 'interaction:\n', est2.summary()
print 'interaction params:\n', est2.params
print "\nConclusion: \nIt's difficult to visually tell the difference in the plot fit lines with or without an interaction term."
print 'Interaction fit has some p-values > 0.05, is it signiicant?  R-squared is the same.  Only F-statistic and Cond. No. seem different.'

plt.clf()
plt.scatter(df2.log_income, df2.int_rate, alpha=0.3, c=df2['home_color'], linewidths=0)
plt.xlabel('Log10 Income')
plt.ylabel('% Interest Rate')
plt.title('Fit: Interaction with home_ownership')

plt.plot(income_linspace, est2.params['Intercept'] + est2.params['home_type'] + est2.params['log_income'] * income_linspace + est2.params['log_income:home_type'] * income_linspace, 'g')
plt.plot(income_linspace, est2.params['Intercept'] + est2.params['home_type'] + 0 * est2.params['log_income'] * income_linspace + 0 * est2.params['log_income:home_type'] * income_linspace, 'b')
plt.plot(income_linspace, est2.params['Intercept'] + est2.params['home_type'] + est2.params['log_income'] * income_linspace + 0 * est2.params['log_income:home_type'] * income_linspace, 'r')
plt.text(4, 2.5, 'green fit line contains all terms')

plt.savefig(plotdir+'intrate_v_income_hometype.png')

'''
Output from fits:

linear (no interaction term):
                            OLS Regression Results                            
==============================================================================
Dep. Variable:               int_rate   R-squared:                       0.014
Model:                            OLS   Adj. R-squared:                  0.014
Method:                 Least Squares   F-statistic:                     3388.
Date:                Thu, 10 Dec 2015   Prob (F-statistic):               0.00
Time:                        21:06:08   Log-Likelihood:            -6.7774e+05
No. Observations:              235628   AIC:                         1.355e+06
Df Residuals:                  235626   BIC:                         1.356e+06
Df Model:                           1                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
Intercept     24.6831      0.188    131.520      0.000        24.315    25.051
log_income    -2.2681      0.039    -58.206      0.000        -2.344    -2.192
==============================================================================
Omnibus:                     6448.193   Durbin-Watson:                   1.967
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             6989.419
Skew:                           0.421   Prob(JB):                         0.00
Kurtosis:                       2.951   Cond. No.                         107.
==============================================================================

with interaction term:
                            OLS Regression Results                            
==============================================================================
Dep. Variable:               int_rate   R-squared:                       0.014
Model:                            OLS   Adj. R-squared:                  0.014
Method:                 Least Squares   F-statistic:                     1134.
Date:                Thu, 10 Dec 2015   Prob (F-statistic):               0.00
Time:                        21:06:14   Log-Likelihood:            -6.7773e+05
No. Observations:              235628   AIC:                         1.355e+06
Df Residuals:                  235624   BIC:                         1.356e+06
Df Model:                           3                                         
========================================================================================
                           coef    std err          t      P>|t|      [95.0% Conf. Int.]
----------------------------------------------------------------------------------------
Intercept               24.3149      0.297     81.972      0.000        23.734    24.896
log_income              -2.1828      0.062    -34.981      0.000        -2.305    -2.061
home_type                0.1824      0.171      1.068      0.285        -0.152     0.517
log_income:home_type    -0.0442      0.036     -1.238      0.216        -0.114     0.026
==============================================================================
Omnibus:                     6453.652   Durbin-Watson:                   1.967
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             6995.766
Skew:                           0.421   Prob(JB):                         0.00
Kurtosis:                       2.951   Cond. No.                         354.
==============================================================================
'''

