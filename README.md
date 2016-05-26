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

The data was randomly split into 75% training data and 25% test data.  We used the training data to model loan behavior, and the test data as an analog for batches of incoming new loan applicants.  Fit and prediction was done comparing several machine learning methods:
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

#### Summary
A summary of the results follows.  
+ Support Vector Machines
    + 90% +- 4% accuracy 
    + insensitive to parameter choice
    + may be slower than other methods
+ Naive Bayes
    + 89% +- 5% accuracy 
    + sensitive to parameter choice (seven columns optimal)
    + reasonably fast
+ Logistic Regression
    + 89% +- 4% accuracy 
    + somewhat sensitive to parameter choice (five to eight columns optimal)
    + fast

Any of the above classification methods will predict high or low interest rate with about 89% +- 5% accuracy.  The training error estimate is low and consistent across methods, indicating the error comes from variability within the data.  *Logistic Regression* is the best choice, since it is fast, accurate, and easy to implement with a little optimization.  

#### Conclusion
Therefore, with reasonable accuracy, we may:
+ target a segment of new customers with extra scrutiny on their loan questionnaires
+ correctly target ads to existing customer segments  

#### Detailed Analysis
A detailed analysis of each method follows.  

##### *Support Vector Machines*
Fit of training data of high or low interest rate from numeric variables was performed using [Support Vector Machine Classification](http://scikit-learn.org/stable/modules/svm.html#svm) with linear kernel and ten-fold [Cross Validation](http://scikit-learn.org/stable/modules/cross_validation.html), scored using prediction accuracy (percent correct).  

Cross-validation can tell us a lot about the data.  Essentially, one may split the training data into subtrain and validation data, fitting the model with, say, 90% subtrain data and testing the prediction accuracy with 10% validation data for a CV factor of 10.  This is repeated 10 times with a different randomly selected validation set, each time giving a new prediction score.   The prediction scores vary with data randomness, and we can do statistics on them to tell how well we fit the model. 

We used the cross-validation prediction scores of the data using SVC to calculate a mean and standard error for different model parameters C.  Their range is shown below in boxplots.  We tested the statistical significance of the scores between different model parameters using a [t-test](https://en.wikipedia.org/wiki/Student%27s_t-test).  Exploration of CV prediction score with an SVC linear kernel showed insensitivity to the C parameter.  

<img src="https://github.com/bfetler/lending_club_predict/blob/master/svm_predict_plots/svm_gridscore_C.png" alt="svm linear_kernel C opt" />

SVC with an rbf kernel showed some sensitivity to C and gamma.  

<img src="https://github.com/bfetler/lending_club_predict/blob/master/svm_predict_plots/svm_gridscore_rbf_gammaC.png" alt="svc rbf_kernel gammaC opt" />

In other words, choice of the C parameter with a linear kernel doesn't matter much, which we can prove by statistics.  For the rbf kernel, there is some significance in choosing C and gamma, but the prediction score is usually lower.  We chose to use the simpler linear kernel with a default value of C = 1.

Using all numeric variables gave an initial score estimate of 87% +- 3% (one standard deviation by cross validation).  Optimization using randomly chosen column variables with CV mean score gave an optimum score of 90% +- 4% with nine columns.   It was somewhat insensitive to parameter choice, with optimum number varying between five and nine columns and scores all around 89% within one standard deviation, indicating the support vectors are somewhat independent of the columns used.  

<img src="https://github.com/bfetler/lending_club_predict/blob/master/svm_predict_plots/svm_opt_params_boxplot.png" alt="optimum parameters boxplot" />

Prediction score of the test data was 90%, within the CV fit score error.  A plot of optimum predicted values is shown below.  

<img src="https://github.com/bfetler/lending_club_predict/blob/master/svm_predict_plots/svm_intrate_optvar_predict.png" alt="optimum prediction" />

The processing script is given in **svm_predict.py**.  Text output is in **svm_predict_output.txt** and plots in **svm_predict_plots/**.  While accurate, SVM methods are known to be slow to compute.

##### *Naive Bayes*
Fit of training data of high or low interest rate from as many as eleven numeric variables was performed using [Gaussian Naive Bayes](http://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes) modeling, scored using fit accuracy.  Using all numeric columns gave a mean CV fit score of 88% +- 5%.  Optimization using randomly chosen column variables gave a best score of 90% +- 5% using only seven variables, with repeat runs giving number of columns varying between four and seven around 89%.  Adding more than seven variables generally decreased the correct prediction rate.  Naive Bayes is known to be sensitive to variable dependence, and it seems not all columns are independent.  

<img src="https://github.com/bfetler/lending_club_predict/blob/master/naive_bayes_plots/gnb_opt_params_boxplot.png" alt="naive bayes optimum parameters boxplot" />

The prediction score of test data was found to be 89%, and a plot is given below.  

<img src="https://github.com/bfetler/lending_club_predict/blob/master/naive_bayes_plots/gnb_intrate_optvar_predict.png" alt="naive bayes predict plots" />

The processing script is given in **naive_bayes.py**.  Text output is in **naive_bayes_output.txt** and plots in **naive_bayes_plots/**.  

##### *Logistic Regression* 
Fit of training data of high or low interest rate from eleven numeric variables was performed using [Logistic Regression](http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression), scored using fit accuracy.  A score of 73% was found without scaling the data, compared to about 90% with scaling.  Exploration of meta-parameters showed insensitivity to C, and an optimum value of C=1 was used.  

<img src="https://github.com/bfetler/lending_club_predict/blob/master/logistic_regression_plots/lr_gridscore_C.png" alt="logistic regression C gridsearch" />

Optimization using randomly chosen column variables with CV gave a best score of 90% +- 4% using eight variables, with optimum number varying between five and eight columns.  

<img src="https://github.com/bfetler/lending_club_predict/blob/master/logistic_regression_plots/lr_opt_params_boxplot.png" alt="logistic regression optimum parameters boxplot" />

Prediction score of test data was estimated at 89%.  A plot is shown below.

<img src="https://github.com/bfetler/lending_club_predict/blob/master/logistic_regression_plots/lr_intrate_optvar_predict.png" alt="logistic regression prediction" />

A processing script is given in **logistic_regression.py**.  Plots of logistic functions are in **logistic_plots/** and script output in **logistic_output.txt**.

#### Conclusion
Any of the above classification methods will predict high or low interest rate with about 89% +- 5% prediction accuracy.
