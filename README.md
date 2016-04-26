## Ad Targeting Using Interest Rate Predictors

We may target different types of questions to loan applicants using variables that predict a high or low interest rate, as a potential predictor of high- or low-risk behavior.  We could also use the same predictors to target advertising to different customer segments.  A model dataset is available from [Lending Club](https://www.lendingclub.com/info/download-data.action), an online lending service.

#### Exploration

The dataset contains 14 variables for 2500 loan applicants from FY 2013, including *Interest.Rate* (the interest rate approved), which may be used as a target variable for supervised learning.  A binary class target *IR_TF* (interest rate true-false) was created, using 0 if below 12% and 1 if above 12%.  After data cleaning, 2498 columns remained.  Histograms gave some indication of data variability.  

<img src="https://github.com/bfetler/lending_club_predict/blob/master/logistic_regression_plots/hist_allvar.png" alt="histograms numeric variables" />

Some histograms showed financial variables were not normally distributed, and were better replaced by log variables.

<img src="https://github.com/bfetler/lending_club_predict/blob/master/logistic_regression_plots/hist_logvar.png" alt="histograms log variables" />

#### Modeling and Prediction

The data was randomly split into 75% training data and 25% testing data.  Fitting and prediction was done comparing several methods:
+ Support Vector Machines
+ Naive Bayes
+ Logistic Regression

The following was done for each method:
+ Initial fit and cross validation of training data.
+ If applicable, optimization of meta-parameters by [grid score with cross validation](http://scikit-learn.org/stable/modules/grid_search.html#grid-search).
+ Variable optimization on training data as follows: 
  + start with two variables *FICO.Score, Amount.Requested*
  + successively add random columns
  + keep columns with improved score (cross validation mean score)
  + repeat N times
+ Finally, prediction on test data using optimized columns and parameters.

#### *svm_predict.py*
Fit of training data of high or low interest rate from eleven numeric variables was performed using [Support Vector Machine Classification](http://scikit-learn.org/stable/modules/svm.html#svm) with linear kernel and ten-fold [Cross Validation](http://scikit-learn.org/stable/modules/cross_validation.html), scored using fit accuracy.  Cross-validation gives scores from each data fold, which may be used to calculate a mean and standard error, giving a measure of data variability.  Their range is shown below in boxplots.  Exploration of SVC meta-parameter scoring with a linear kernel showed insensitivity to C.

<img src="https://github.com/bfetler/lending_club_predict/blob/master/svm_predict_plots/svm_gridscore_C.png" alt="svm linear_kernel C opt" />

SVC with an rbf kernel showed some sensitivity to C and gamma.  

<img src="https://github.com/bfetler/lending_club_predict/blob/master/svm_predict_plots/svm_gridscore_rbf_gammaC.png" alt="svc rbf_kernel gammaC opt" />

A linear kernel with optimum value of C=1 was chosen.  Using all numeric variables gave an initial score estimate of 87% +- 3% (one standard deviation by cross validation).  Optimization using randomly chosen column variables with CV mean score gave an optimum score of 90% +- 4% with nine columns.   It was somewhat insensitive to parameter choice, with optimum number varying between five and nine columns and scores all around 89% within a standard deviation, indicating the support vectors are somewhat independent of the columns used.  

<img src="https://github.com/bfetler/lending_club_predict/blob/master/svm_predict_plots/svm_opt_params_boxplot.png" alt="optimum parameters boxplot" />

Prediction score of the test data was 90%, within the CV fit score error.  A plot of optimum predicted values is shown below.  

<img src="https://github.com/bfetler/lending_club_predict/blob/master/svm_predict_plots/svm_intrate_optvar_predict.png" alt="optimum prediction" />

Text output is given in **svm_predict_output.txt** and plots in **svm_predict_plots/**.  While accurate, SVM methods are known to be slow to compute.

#### *naive_bayes.py*
Fit of training data of high or low interest rate from as many as eleven numeric variables was performed using [Gaussian Naive Bayes](http://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes) modeling, scored using fit accuracy.  Using all numeric columns gave a mean CV fit score of 88% +- 5%.  Optimization using randomly chosen column variables gave a best score of 90% +- 5% using only seven variables, with repeat runs giving number of columns varying between four and seven around 89%.  Adding more than seven variables generally decreased the correct prediction rate.  Naive Bayes is known to be sensitive to variable dependence, and it seems not all columns are independent.  

<img src="https://github.com/bfetler/lending_club_predict/blob/master/naive_bayes_plots/gnb_opt_params_boxplot.png" alt="naive bayes optimum parameters boxplot" />

The prediction score of test data was found to be 89%, and a plot is given below.  

<img src="https://github.com/bfetler/lending_club_predict/blob/master/naive_bayes_plots/gnb_intrate_optvar_predict.png" alt="naive bayes predict plots" />

Text output is given in **naive_bayes_output.txt** and plots in **naive_bayes_plots/**.  

#### *logistic_regression.py* 
Fit of training data of high or low interest rate from eleven numeric variables was performed using [Logistic Regression](http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression), scored using fit accuracy.  A score of 73% was found without scaling the columns, compared to about 90% with scaling.  Exploration of meta-parameters showed insensitivity to C, and an optimum value of C=1 was used.  

<img src="https://github.com/bfetler/lending_club_predict/blob/master/logistic_regression_plots/lr_gridscore_C.png" alt="logistic regression C gridsearch" />

Optimization using randomly chosen column variables with CV gave a best score of 89% +- 4% using seven variables, with optimum number varying between five and seven columns.  

<img src="https://github.com/bfetler/lending_club_predict/blob/master/logistic_regression_plots/lr_opt_params_boxplot.png" alt="logistic regression optimum parameters boxplot" />

Prediction score of test data was estimated at 88%.  A plot is shown below.

<img src="https://github.com/bfetler/lending_club_predict/blob/master/logistic_regression_plots/lr_intrate_optvar_predict.png" alt="logistic regression prediction" />

Plots of logistic functions are in **logistic_plots/** and script output in **logistic_output.txt**.

#### Conclusion
Any of the above classification methods will predict high or low interest rate with about 89% accuracy.  The training error estimate ~5% is low and consistent across methods, indicating the error comes from variability within the data.  Therefore we may target loan questionnaires and ads to customers with reasonable accuracy.  
