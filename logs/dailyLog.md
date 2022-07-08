## Week 0: Reading Literature

### Original Work by Dr. Paul Houston, Apurba Nandi, and Dr. Joel Bowman
* [A Machine Learning Approach for Prediction of Rate Constants](https://pubs.acs.org/doi/10.1021/acs.jpclett.9b01810)
* [A Machine Learning Approach for Rate Constants. II. Clustering, Training, and Predictions for the O(3P) + HCl → OH + Cl Reaction](https://pubs.acs.org/doi/10.1021/acs.jpca.0c04348)


## Week 1: Getting Familiar with Data, Initial Models

### June 13, 2022 (Day 1)
Updates:
* Began looking at the chi dataset. The data.dat file contains 404 data instances. There are five variables, namely ustat, alph1, alph2, beta, and chi.  For the problem we want to solve, ustat, alph1, alph2, and beta are features, while chi is our target.
* Examined the relationship with each of the features with the target Chi. The goal of this is to see if there are any patterns in the data. Relationships were plotted on a scatter plot with chi on the y-axis and given feature on the x-axis. There does not appear to be an immediate relationship among features and target. 
* Examined and evaluated the distribution of data across all variables. The goal of this is to efficiently find ways to bin the continuous data, so these 'bins' can be used as categorical data in later steps (i.e. when data is processed by classification algorithms). 
* Houston and group created two labels, small chi and large chi, for the target 'chi.' The 'boundary' separating these two clusters was found by eyeing the data; the boundary noted in their [work](https://pubs.acs.org/doi/10.1021/acs.jpca.0c04348) is located at chi = 3. When chi is less than 3, the data instance is labeled as "small chi." When chi is greater than or equal to 3, the data instance is labeled as "large chi." 
* I wanted to see whether there was an alternate (and possible more systematic) way of finding this decision boundary. Hence, I used [DBSCAN from Scikit Learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html). Density-based spatial clustering of applications with noise or DBSCAN is a density-based clustering non-parametric algorithm: given a set of points in some space, it groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions (whose nearest neighbors are too far away).
    - Resources:
        1. [DBSCAN Wikipedia](https://en.wikipedia.org/wiki/DBSCAN)
        2. [DBSCAN from Scikit Learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
* Setting the epsilon value to 1 (local radius) and the number of samples to six, we achieve a scatterplot that closely resembles having a boundary line at chi = 3. However, it is important to note that while blue represents one cluster, red represents outliers when we are using DBSCAN. 
* Results and notes can be found on [Week01 - Visualizing Chi Data, Finding Patterns.ipynb](../nb/Week01/Week01%20-%20Models.ipynb). 

### June 14, 2022 (Day 2)
Updates:
* Obtained Small_chi.dat and Large_chi.dat from Dr. Bowman. There are the datasets created by Houston and group in their [work](https://pubs.acs.org/doi/10.1021/acs.jpca.0c04348). When both datasets are plotted on the same plane, we see the separation at chi = 3. 
* Ideally, we want to solve this as a binary classification problem. Data preprocessing was performed as we made changes to existing data based on examining the boxplot and the measures of central tendency from [Week01 - Visualizing Chi Data, Finding Patterns](../nb/Week01/Week01%20-%20Models.ipynb). Continuous data was converted to categorical, nominal data:
    - Chi: created binary labels: 0 = small chi (chi < 3), 1 = large chi (chi >= 3)
    - Ustat: created binary labels: 0 = (ustat < 3), 1 = (ustat >= 3)
    - Alph1: created labels: 0 = (alph1 < 1), 1 = ( 1 <= alph1 < 2), 2 = (alph1 >= 2)
    - Alph2: created binary labels: 0 = (alph2 < 2), 1 = (alph2 >= 2)
    - Beta: created binary labels: 0 = (beta < 50), 1 = (beta >= 50)
* Results and notes can be found on [Week01 - Preprocessing.ipynb](../nb/Week01/Week01%20-%20Preprocessing.ipynb).
* The new dataset can be found [here](../data/processed/week01.csv).

### June 15, 2022 (Day 3)
Updates:
* Ran four classification algorithms on the new data: logistic regression (LR), gaussian naive bayes (GB), random forest (RF), and adaboost (AB). 
* For model assessment, I used train test split (from Scikit Learn), AUC curve, and confusion matrix. For the confusion matrix: 0 (small chi) is negative and 1 (large chi) is positive. 
* Results are summarized below: 
    - LR:
        - Train Score: 0.91
        - Test Score: 0.89 
        - Training Confusion Matrix: TN = 247, FP = 0, FN = 23, TP = 0
        - Testing Confusion Matrix: TN = 119, FP = 0, FN = 15, TP = 0
        - AUC: 0.73
    - GB: 
        - Train Score: 0.60
        - Test Score: 0.54
        - Training Confusion Matrix: TN = 138, FP = 109, FN = 0, TP = 23
        - Testing Confusion Matrix: TN = 57, FP = 62, FN = 0, TP = 15
        - AUC: 0.75
    - RF:
        - Train Score: 0.91
        - Test Score: 0.89 
        - Training Confusion Matrix: TN = 247, FP = 0, FN = 23, TP = 0
        - Testing Confusion Matrix: TN = 119, FP = 0, FN = 15, TP = 0
        - AUC: 0.80
    - AB:
        - Train Score: 0.91
        - Test Score: 0.89 
        - Training Confusion Matrix: TN = 247, FP = 0, FN = 23, TP = 0
        - Testing Confusion Matrix: TN = 119, FP = 0, FN = 15, TP = 0
        - AUC: 0.69
* For LR, RF, and AB, the confusion matrices display only results for TN and FN. This means that the models only predicts small chi for all data instances. This may be a result of having a large disproportion of small chi to large chi data. 
* For GB: The test score was very low. The model does attempt to identify large chi (shown through TP and FP), but often incorrectly identifies small chi (shown through large FP value). Issues may occur because the data does not follow a Gaussian distribution. 
* Results and notes can be found on [Week01 - Models.ipynb](../nb/Week01/Week01%20-%20Models.ipynb).

## Week 2: Understanding Results and Finding Alternatives

### June 21, 2022 (Day 4)
* Issue: We have low True Positive rate in our models. Our models are not predicting one of the two target classes. 
* [How to increase true positive in your classification Machine Learning model?](https://stackoverflow.com/questions/58074203/how-to-increase-true-positive-in-your-classification-machine-learning-model)
    - You can change your model and test whether it performs better or not
    - You can Fix a different prediction threshold : here I guess you predict 0 if the output of your regression is <0.5, you could change the 0.5 into 0.25 for example. It would increase your True Positive rate, but of course, at the price of some more False Positives.
    - You can duplicate every positive example in your training set so that your classifier has the feeling that classes are actually balanced.
    - You could change the loss of the classifier in order to penalize more False Negatives (this is actually pretty close to duplicating your positive examples in the dataset)
    - sklearn.ensemble.GradientBoostingClassifier
* [How do I increase the true positive in my classification machine learning model?](https://www.quora.com/How-do-I-increase-the-true-positive-in-my-classification-machine-learning-model)
    - Maybe your model needs more examples of true positives to learn how to classify them. Then you need to supply more data to your model.
    - You can apply some preprocessing to your data. E.g. mean normalization, removing outliers, etc.
    - Sometimes the architecture that you’re using may not be optimal. For instance, instead of coding a neural network from scratch, you can try transfer learning. Finetuning the parameters can also be helpful. Cross-Validation/K-Fold validation will help you to decide, which of the models/set of parameters works best for your task.
    - Last, but not least, there’re some fancy techniques like progressive resizing or stacking models, that are likely to improve your overall score.
    - Increasing True Positive is directly linked with the predictive model’s capability in identifying Positive cases correctly.
        - Data Imbalance: There are very few rows of Positive cases as compared to Negative cases (eg. 10 rows for Positive and 90 rows for Negative)
        - Solution: Oversample the Positive cases and force the algorithm to learn how to correctly identify positives.
        - Bad Data: This happens when the predictors which you are using, does not have any correlation with your target variable.
        - Solution: Try using different predictors Or use some transformations on the existing predictors (e.g. log, sqrt, standardization, etc.)

### June 22, 2022 (Day 5)     
* How can we uncover further patterns from the data, beyond the target chi? 
* Explored relationships between features and other features. Wanted to see if any notable clusters can be made from them. Unfortunately, there are no easy way to group data instances in these scatters. 
* Alternatively, I used the chi labeling (small chi and large chi) used by Houston and group to color code the data instances, and replotted the scatters. Some visual observations I made:
    - Large chi appears in a wide range of alph1 and beta values, but not alph2 values. Large chi only appears when alph2 is closer to zero.
    - Following the above observation, large x2-values do not correspond to increased large-chi values.
* How can we restructure labels for chi? 
* Wanted to explore if there was another way to group together data instances, beyond DBSCAN. The K-means clustering algorithm computes centroids and repeats until the optimal centroid is found. It is presumptively known how many clusters there are. K-clusters for N instances.
     - Resources:
        1. [KMeans Wikipedia](https://en.wikipedia.org/wiki/K-means_clustering)
        2. [KMeans from Scikit Learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
* Unfortuntately, we were not able to achieve good clusters from the scatter of ustat v chi. Clusters were split near ustat = 5. 

### June 25, 2022 (Day 6)  
* Presented [PowerPoint](/files/Week%2001%20and%2002%20Report.pdf) for Week 1 and Week 2 results.
* Discussed ways to move on and potential ways to improve results. 
    - Adding an additional feature should aid in problems with low TP and FP by providing more data to work with. The new feature suggested is the absolute difference between alph1 and alph2. 
    - We should see if there is some way to distinguish data instances in the 'overlapping' region in ustat v chi.
    - Adding noise can prevent overfitting of data.  

## Week 3: Improving Models and Feature Engineering

* Results and notes for the week can be found on [Week03 - Model Assessments](../nb/Week03/Week03%20-%20Model%20Assessments.ipynb) and [Week03 - Visualization and Feature Engineering](../nb/Week03/Week03%20-%20Visualization%20and%20Feature%20Engineering.ipynb).

### June 28, 2022 (Day 7)  
* [Should I convert a continuous variable to a categorical variable?](https://argoshare.is.ed.ac.uk/healthyr_book/should-i-convert-a-continuous-variable-to-a-categorical-variable.html)
    - The clear disadvantage in doing this is that information is being thrown away. Which feels like a bad thing to be doing. This is particularly important if the categories being created are large.
    - It is unforgivable practice to repeatedly try different cuts of a continuous variable to obtain a statistically significant result. 
    - When communicating the results of an analysis to a lay audience, it may be easier to use a categorical representation.
    - TLDR: Do not do it unless you have to. Plot and understand the continuous variable first. If you do it, try not to throw away too much information. Repeat your analyses both with the continuous data and categorical data to ensure there is no difference in the conclusion (often called a sensitivity analysis).
* Converted the categorical features back into continuous features to preserve important info that may have been lost. Features remain continuous from now on, unless otherwise stated. The target variable has been left the same, however. 
* The new dataset can be found [here](../data/processed/week03_nobin.csv).
* Followed the same procedures as [Week01 - Models.ipynb](../nb/Week01/Week01%20-%20Models.ipynb) in processing and assessing data, except this time we relied only on Logistic Regression as our go-to algorithm. 
* The results are pretty good with high training accuracy of 0.95 and testing accuracy of 0.92. The area under the curve score is high with a score of 0.95. The confusion matrix shows that the model does attempt to identify both groups, with a good True Negative (small chi) and True Positive (large chi).

### June 29, 2022 (Day 8)  
* Adding a new feature may potentially help in improving model accuracy by providing more useful information and speeding up data transformation.
* Created a new dataset with 'diff' as a new feature. Diff is the absolute difference between alph1 and alph2. Diff remains a feature unless stated otherwise. Thus, there is now a total of five features. 
* The new dataset can be found [here](../data/processed/week03_nobin_diff.csv).
* Followed the same procedures as before and used Logistic Regression. There were good scores for True Negative and True Positive as the models attempts to predict small and large chi with some errors. 
* Removed alph1 and alph2 from dataset, because they may be redundant data. Thus, there is only three features. Performed the same procedures and model assessment. The model perfectly predicts small chi and large chi with no errors. The AUC curve is perfect. 
* High AUC and perfect AUC may be the result of overfitting, stemming from a small dataset. Will need to see if it is possible to generate reasonable artificial data and/or noise to dataset in later steps.  

### June 30, 2022 (Day 9)  
* Bowman suggested: 
    - In addition to try SVM using u* and alphdif =alpha1-alpha2 I had a thought about RF. Suppose you use u* and alphdif as inputs again.  Then train on all data in the large chi cluster and a random split (maybe 50:50) of the small chi cluster.  Then test on the data left out of the small chi cluster.  What we want to see if that test data gets assigned to the small cluster with no errors.  I would expect some errors, i.e., assigned to the large chi cluster.  But if this works this could be another option for assigning clusters to new data.
* Goal: See if test data gets assigned to small chi with no errors. 
* Divided data as follows: 
    - Training data consists of 183 random instances of small chi, all (38) instances of large chi.
    - Testing data consists of the remaining 183 random instances of small chi.
    - Kept all features. 
* The new dataset can be found [here](../data/processed/week03_train_df_sugg01.csv) and [here](../data/processed/week03_test_df_sugg01.csv).
* Confusion matrix shows 174 instances were correctly identified as small chi (True Negative) and 9 were incorrectly identified as 'large chi (False Positive). As expected, the model does not predict perfectly. 

### July 1, 2022 (Day 10)  
* Bowman suggested: 
    - Given the ustat v chi scatter, can you cut from the small and large chi datasets just the data with u* between 5 and 12?  Then we can eyeball the these datasets and see if we spot anything that might distinguish the two.
* Goal: Process only data instances that lie between 5.26 and 12.6. The minimum ustat value for large chi is 5.26 and the maximum ustat value for small chi is 12.6. 
* There are a total of 128 data instances: 98 small chi and 30 large chi.
* The new dataset can be found [here](../data/processed/week03_overlap_data.csv).
* Same process and model assessment as before. There is good testing accuracy and AUC curve. Model predicts small and large chi with errors. Issues may stem from not having enough testing data for large chi.

## Week 4: Improving Models and Feature Engineering

### July 5, 2022 (Day 11)  
* Presented [PowerPoint](/files/Week%2003%20Report.pdf) for Week 3 results.
* Discussed results and future direction.
    - Since data is expensive, it is plausible to create artificial data to use for testing. The data just has to be reasonable and 'predictable' (i.e. we are sure of the classification). 
    - Retry processing and assessment but remove alph1 and alph2 as features.
    - Examining the 'overlap' data further, we want to be able to classify this region. Does not have to be small chi or large chi. 




https://python-course.eu/machine-learning/artificial-datasets-with-scikit-learn.php