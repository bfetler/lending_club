## Ad Targeting Using Interest Rate Predictors

We may target different types of questions to loan applicants using variables that predict a high or low interest rate, as a potential predictor of high- or low-risk behavior.  We could also use the same predictors to target advertising to different customer segments.  A model dataset is available from [Lending Club](https://www.lendingclub.com/info/download-data.action), an online lending service.

#### Exploration

Data exploration is given in __lending_club_explore.py.__  The dataset contains 14 variables for 2500 loan applicants from FY 2013, including *Interest.Rate*, the interest rate approved, which may be used as a target variable for supervised learning.  A binary class target *IR_TF* (interest rate true false) was created, with 0 if below 12% and 1 above 12%.  After data cleaning, 2498 columns remained.  

A scatter matrix revealed some correlations between the eleven numeric variables.  Histograms showed that some financial variables were not normally distributed, and were replaced by log variables.  

The data was randomly split into 75% training and 25% testing data.

#### Modeling and Prediction

Fitting and prediction was done comparing several methods:
+ Support Vector Machines
+ Naive Bayes
+ Logistic Regression

The following was done for each method:
+ Initial fit and cross validation of training data.
+ If applicable, optimization of meta-parameters by grid score with cross validation.
+ An initial prediction on the test data.
+ Variable optimization on training data as follows: 
  + start with two variables *FICO.Score, Amount.Requested*
  + successively add random columns
  + keep those with improved score (cross validation mean)
  + repeat N times
+ Finally, prediction on test data using optimized columns and parameters.

#### *svm_predict.py*
Fit of training data of high or low interest rate from eleven numeric variables was performed using [Support Vector Machine Classification](aaa) with linear kernel and ten-fold [Cross Validation](ccc), scored using CV mean fit score accuracy.   SVC parameter scoring with a linear kernel showed insensitivity to C, and an optimum value of C=1 was chosen.  Using all numeric variables gave an initial score estimate of 89% +- 3% with standard error by cross validation.  Optimization using randomly chosen column variables using CV mean score was somewhat insensitive to parameter choice, with optimum number varying between five and nine columns, indicating the support vectors are somewhat independent of the columns used.  Prediction score of test data was within the CV fit score error.  Text output is given in **svm_predict_output.txt** and plots in **svm_predict_plots/**.  While accurate, SVM methods are known to be slow to compute.

#### *naive_bayes.py*
Fit of training data of high or low interest rate from as many as eleven numeric variables was performed using [Gaussian Naive Bayes](bbb) modeling, scored using fit accuracy.  Optimization using randomly chosen column variables with CV gave a best score of 89% +- 4% using only five variables.  Adding more variables generally decreased the correct prediction rate.  Naive Bayes is known to be sensitive to variable dependence, and it seems not all columns are independent.  Text output is given in **naive_bayes_output.txt** and plots in **naive_bayes_plots/**.  

#### *logistic_regression.py* 
Fit of training data of high or low interest rate from eleven numeric variables was performed using [Logistic Regression](aaa), scored using fit accuracy.  A score of 70% was found without scaling the columns, compared to ~90% with scaling.  Optimization using randomly chosen column variables with CV gave a best score of 89% using seven variables, with optimum number varying between six and nine columns.  Plots of logistic functions are in **logistic_plots/** and script output in **logistic_output.txt**.


