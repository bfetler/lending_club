## Loan Scrutiny and Ad Targeting Using Interest Rate Predictors

Can we use consumer loan data to assess the level of scrutiny needed for future loan applicants?  For example, we could target different types of questions to loan applicants using variables that predict a high or low interest rate, as a potential predictor of high- or low-risk behavior.  

Could we also use loan data to predict which types of ads consumers should receive?  We could potentially use the same predictors to target advertising to different customer segments.  

A consumer loan dataset is available from [Lending Club](https://www.lendingclub.com/info/download-data.action), an online lending service, which we may use to explore these questions.

#### Exploration

The dataset contains 14 variables for 2500 loan applicants from FY 2013, including *Interest.Rate* (the interest rate approved), which may be used as a target variable for supervised learning.  We divided consumers based on a target interest rate into two categories, *high interest* if above 12% and *low interest* if below 12%, using a variable *IR_TF* (interest rate true-false).  After data cleaning, 2498 columns remained.  Histograms gave some indication of data variability.  

<img src="https://github.com/bfetler/lending_club_predict/blob/master/logistic_regression_plots/hist_allvar.png" alt="histograms numeric variables" />

Some histograms of financial variables were not normally distributed, and were replaced by log variables for statistical analysis.

<img src="https://github.com/bfetler/lending_club_predict/blob/master/logistic_regression_plots/hist_logvar.png" alt="histograms log variables" />

#### Modeling and Prediction

The data was randomly split into 75% training data and 25% test data.  We used the training data to model loan behavior, and the test data as an analog for batches of incoming new loan applicants.  Fit and prediction was done comparing several machine learning methods.  The following was done for each method:
+ Initial fit and cross validation of training data.
+ Optimization of meta-parameters by [grid search with cross validation](http://scikit-learn.org/stable/modules/grid_search.html#grid-search) if desired.
+ Cross validation with statistics tells us whether or not further parameter optimization is needed.
+ Variable optimization on training data as follows: 
  + start with two variables *FICO.Score, Amount.Requested*
  + successively add random columns
  + keep columns with improved score (cross validation mean score)
  + repeat N times
+ Finally, prediction on test data using optimized columns and parameters.

#### Summary
A summary of the results follows.  
+ Support Vector Machines
    + 90% +- 4% accuracy 
    + insensitive to parameter choice
    + slightly slower than other methods
+ Naive Bayes
    + 89% +- 5% accuracy 
    + sensitive to parameter choice (seven columns optimal)
    + reasonably fast
+ Logistic Regression
    + 89% +- 4% accuracy 
    + somewhat sensitive to parameter choice (five to eight columns optimal)
    + fast

Any of the above classification methods will predict high or low interest rate with about 89% +- 5% accuracy.  The training error estimate is low and consistent across methods, indicating the error comes from variability within the data.  *Logistic Regression* is the best choice, since it is fast, accurate, and easy to implement with a little optimization.  

#### Detailed Analysis by Logistic Regression
A detailed analysis by Logistic Regression follows.  

Fit of training data of high or low interest rate from eleven numeric variables was performed using [Logistic Regression](http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression), scored using fit accuracy.  A score of 90% was found after scaling the data.  

Cross-validation can tell us whether or not further parameter optimization is needed.  Essentially, by splitting the training data into subtrain and validation data, and fitting the model with a CV factor of 10 (90% subtrain and 10% test data), one may repeat the process 10 times with a slightly different random data set.  Each fold gives a new prediction score, and one may do statistics on the scores to tell how well we fit the model.   

We used cross-validation prediction scores of the data from Logistic Regression to calculate a mean and standard error for different model parameters.  Their range is shown below in boxplots.  We tested the statistical significance of the scores between different model parameters using a [t-test](https://en.wikipedia.org/wiki/Student%27s_t-test), which shows insensitivity to C at higher values.  __*The error bars are bigger than the variation in accuracy for most values.*__  

In other words, __*the choice of C doesn't matter much*__ as long as the value is high enough, which we can __*measure by statistics*__.  We chose the standard value of C=1 for our model.  

<img src="https://github.com/bfetler/lending_club_predict/blob/master/logistic_regression_plots/lr_gridscore_C.png" alt="logistic regression C gridsearch" />

Optimization using randomly chosen column variables with CV gave a best score of 90% +- 4% using eight variables, with optimum number varying between five and eight columns.  

<img src="https://github.com/bfetler/lending_club_predict/blob/master/logistic_regression_plots/lr_opt_params_boxplot.png" alt="logistic regression optimum parameters boxplot" />

Prediction score of test data was estimated at 89%.  A plot is shown below.

<img src="https://github.com/bfetler/lending_club_predict/blob/master/logistic_regression_plots/lr_intrate_optvar_predict.png" alt="logistic regression prediction" />

A processing script is given in **logistic_regression.py**.  Plots of logistic functions are in **logistic_regression_plots/** and script output in **logistic_regression_output.txt**.

#### Interest Rate by Linear Regression

To predict the actual interest rate from other variables, rather than just whether the interest rate was high or low, we applied Linear Regression.  Fitting the training data using all columns gave an accuracy of 76% +- 5% by cross-validation, while prediction of test data gave 76% accuracy.  By examining variable importance, we found we could model the same fitting accuracy using only five of the columns plus the Intercept:
+ FICO.Score
+ Loan.Length
+ Amount.Funded.By.Investors
+ Inquiries.in.the.Last.6.Months
+ Log.CREDIT.Lines

A processing script is given in **linear_regression.py**.  Plots of logistic functions are in **linear_regression_plots/** and script output in **linear_regression_output.txt**.  Interest rate prediction is shown below.  

<img src="https://github.com/bfetler/lending_club/blob/master/linear_regression_plots/predict_scatter_test.png" alt="linear regression prediction" />

#### Conclusion
We can predict high or low interest rate with about 89% +- 5% accuracy.  Therefore, we may:
+ accurately target new customer segments with extra scrutiny on their loan questionnaires
+ correctly target ads to existing customer segments  
