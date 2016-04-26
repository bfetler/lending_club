## Ad Targeting Using Interest Rate Predictors

We may target different types of questions to loan applicants using variables that predict a high or low interest rate, as a potential predictor of high- or low-risk behavior.  We could also use the same predictors to target advertising to different customer segments.  A model dataset is available from [Lending Club](https://www.lendingclub.com/info/download-data.action), an online lending service.

#### Exploration

Data exploration is given in __lending_club_explore.py.__  The dataset contains 14 variables for 2500 loan applicants from FY 2013, including *Interest.Rate* (the interest rate approved), which may be used as a target variable for supervised learning.  A binary class target *IR_TF* (interest rate true-false) was created, using 0 if below 12% and 1 if above 12%.  After data cleaning, 2498 columns remained.  

A scatter matrix revealed some correlations between the eleven numeric variables.  Histograms showed that some financial variables were not normally distributed, and were replaced by log variables.  

The data was randomly split into 75% training and 25% testing data.

#### Modeling and Prediction

Fitting and prediction was done comparing several methods:
+ Support Vector Machines
+ Naive Bayes
+ Logistic Regression

The following was done for each method:
+ Initial fit and cross validation of training data.
+ If applicable, optimization of meta-parameters by [grid score with cross validation](http://scikit-learn.org/stable/modules/grid_search.html#grid-search).
+ An initial prediction on the test data.
+ Variable optimization on training data as follows: 
  + start with two variables *FICO.Score, Amount.Requested*
  + successively add random columns
  + keep columns with improved score (cross validation mean score)
  + repeat N times
+ Finally, prediction on test data using optimized columns and parameters.

#### *svm_predict.py*
Fit of training data of high or low interest rate from eleven numeric variables was performed using [Support Vector Machine Classification](http://scikit-learn.org/stable/modules/svm.html#svm) with linear kernel and ten-fold [Cross Validation](http://scikit-learn.org/stable/modules/cross_validation.html), scored using CV mean fit score accuracy.  Exploration of SVC meta-parameter scoring with a linear kernel showed insensitivity to C, while the rbf kernel showed some sensitivity to C and gamma.  A linear kernel with optimum value of C=1 was chosen.  Using all numeric variables gave an initial score estimate of 87% +- 3% (one standard deviation by cross validation).  Optimization using randomly chosen column variables with CV mean score gave an optimum score of 90% +- 4% with nine columns.   It was somewhat insensitive to parameter choice, with optimum number varying between five and nine columns and scores all around 89% within a standard deviation, indicating the support vectors are somewhat independent of the columns used.  Prediction score of the test data was 90%, within the CV fit score error.  Text output is given in **svm_predict_output.txt** and plots in **svm_predict_plots/**.  While accurate, SVM methods are known to be slow to compute.

#### *naive_bayes.py*
Fit of training data of high or low interest rate from as many as eleven numeric variables was performed using [Gaussian Naive Bayes](http://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes) modeling, scored using fit accuracy.  Using all numeric columns gave a mean CV fit score of 88% +- 5%.  Optimization using randomly chosen column variables gave a best score of 90% +- 5% using only seven variables, with repeat runs giving number of columns varying between four and seven around 89%.  Adding more than seven variables generally decreased the correct prediction rate.  Naive Bayes is known to be sensitive to variable dependence, and it seems not all columns are independent.  The prediction score of test data was found to be 89%.  Text output is given in **naive_bayes_output.txt** and plots in **naive_bayes_plots/**.  

#### *logistic_regression.py* 
Fit of training data of high or low interest rate from eleven numeric variables was performed using [Logistic Regression](http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression), scored using fit accuracy.  A score of 73% was found without scaling the columns, compared to ~90% with scaling.  Exploration of meta-parameters showed insensitivity to C, and an optimum value of C=1 was used.  Optimization using randomly chosen column variables with CV gave a best score of 89% +- 4% using seven variables, with optimum number varying between six and nine columns.  Prediction score of test data was estimated at 88%.  Plots of logistic functions are in **logistic_plots/** and script output in **logistic_output.txt**.
