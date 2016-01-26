# lending_club

**Regression Algorithm Tests** in Python using loan data from [Lending Club](https://www.lendingclub.com/info/download-data.action), an online lending service.

#### *prob_lending_club.py* 
Boxplot, Histogram, QQplot of *Amount.Funded* and *Amount.Requested* from loan data subset.  Plots are in **univariate/**.
#### *chi_squared.py* 
Chi-squared plot of *Open.Credit.Lines* from loan data subset.  Plots are in **chisq_plots/**.
#### *linear_regression.py* 
Ordinary Least Squares (OLS) Regression of *Interest.Rate* vs. *Amount.Requested* and *FICO.Average* of loan data subset.  Plots of Scatter Matrix and Histograms are in **linear_plots/**.
#### *logistic_regression.py* 
Logistic Regression of *Interest.Rate < 12%* using *Amount.Requested* and *FICO.Average* of loan data subset.  Plots of logistic function are in **logistic_plots/**.
#### *multivariate.py* 
Multivariate Regression of *int_rate* (interest rate) using *log_income* (log annual income) and *home_ownership*.  Data is from the full lending club loan data set for years 2013 and 2014.  Multivariate plots are in **multivar_plots/**.
#### *time_series.py*
Time series ARIMA analysis (autoregression integrated moving average) of monthly loan count for 2013 and 2014.  *ARIMA(1, 1, 0)* seems to best model the data without overfitting.  Plots are in **time_series_plots/** and script output is written to **time_series_output.txt**.
