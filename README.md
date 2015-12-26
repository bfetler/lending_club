# lending_club

#### *prob_lending_club.py* 
Boxplot, Histogram, QQplot of *Amount.Funded* and *Amount.Requested* loan data.  Plots are in **univariate/**.
#### *chi_squared.py* 
Chi-squared plot of *Open.Credit.Lines*.  Plots are in **chisq_plots/**.
#### *linear_regression.py* 
Ordinary Least Squares (OLS) Regression of *Interest.Rate* vs. *Amount.Requested* and *FICO.Average*.  Plots of Scatter Matrix and Histograms are in **linear_plots/**.
#### *logistic_regression.py* 
Logistic Regression of *Interest.Rate < 12%* using *Amount.Requested* and *FICO.Average*.  Plots of logistic function are in **logistic_plots/**.
#### *multivariate.py* 
Multivariate Regression of *int_rate* (interest rate) using *log_income* (log annual income) and *home_ownership*.  Data is from the full lending club 2013-2014 loan data set.  Multivariate plots are in **multivar_plots/**.
#### *time_series.py*
Time series ARIMA analysis (autoregression integrated moving average).  *ARIMA(1, 0, 0)* seems to best model the data without overfitting.  Plots are in **time_series_plots/** and script output is written to **time_series_output.txt**.
