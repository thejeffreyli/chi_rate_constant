## Week 0: Reading Literature

### Original Work by Dr. Paul Houston, Apurba Nandi, and Dr. Joel Bowman
* [A Machine Learning Approach for Prediction of Rate Constants](https://pubs.acs.org/doi/10.1021/acs.jpclett.9b01810)
* [A Machine Learning Approach for Rate Constants. II. Clustering, Training, and Predictions for the O(3P) + HCl → OH + Cl Reaction](https://pubs.acs.org/doi/10.1021/acs.jpca.0c04348)


## Week 1: Getting Familiar with Data, Initial Models

### June 13, 2022 (Day 1)
Updates:
* Began looking at the chi dataset. The data.dat file contains 404 data instances. There are five variables, namely ustat, alph1, alph2, beta, and chi.  For the problem we want to solve, ustat, alph1, alph2, and beta are features, while chi is our target.
* Examined the relationship with each of the features with the target Chi. The goal of this is to see if there are any patterns in the data. Relationships were plotted on a scatter plot with chi on the y-axis and given feature on the x-axis. There does not appear to be an immediate relationship among features and target. 
* Examined and evaluated the distribution of data across all variables. Since I will be running classification algorithms (and ultimately developing a model), I wanted to bin the continuous data to create usable and easier to understand categorical data. 
* Houston and group created two labels, small chi and large chi, for the target 'chi.' The 'boundary' separating these two clusters was found by eyeing the data; the boundary noted in their [work](https://pubs.acs.org/doi/10.1021/acs.jpca.0c04348) is located at chi = 3. When chi is less than 3, the data instance is labeled as "small chi." When chi is greater than or equal to 3, the data instance is labeled as "large chi." 
* I wanted to see whether there was an alternate (and possible more systematic) way of finding this decision boundary. Hence, I used [DBSCAN from Scikit Learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html). Density-based spatial clustering of applications with noise or DBSCAN is a density-based clustering non-parametric algorithm: given a set of points in some space, it groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions (whose nearest neighbors are too far away).
    - Resources:
        1. [DBSCAN Wikipedia](https://en.wikipedia.org/wiki/DBSCAN)
        2. [DBSCAN from Scikit Learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
* Setting the epsilon value to 1 (local radius) and the number of samples to six, we achieve a scatterplot that closely resembles having a boundary line at chi = 3. However, it is important to note that while blue represents one cluster, red represents outliers when we are using DBSCAN. 



Results and notes can be found on Week01 - Visualizing Chi Data, Finding Patterns. 
### June 14, 2022 (Day 2)
Updates:
* Obtained Small_chi.dat and Large_chi.dat from Dr. Bowman. These are prelabeled data instances. When displayed on a scatterplot, we see that they are separated by a decision boundary at chi = 3. 
* Ideally, we want to be able to solve this as a binary classification problem. I figured it would be best to perform some feature engineering with the existing data. Changes made to the data were based on examining the boxplot and the measures of central tendency fom Day 1. I summarized the changes:
    - Chi: created binary labels: 0 = small chi (chi < 3), 1 = large chi (chi >= 3)
    - Ustat: created binary labels: 0 = (ustat < 3), 1 = (ustat >= 3)
    - Alph1: created labels: 0 = (alph1 < 1), 1 = ( 1 <= alph1 < 2), 2 = (alph1 >= 2)
    - Alph2: created binary labels: 0 = (alph2 < 2), 1 = (alph2 >= 2)
    - Beta: created binary labels: 0 = (beta < 50), 1 = (beta >= 50)
* The new dataset can be found in data file. 

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
* For GB: The test score was very low. The model does attempt to identify large chi (shown through TP and FP), but often incorrectly identifies small chi (shown through large FP value).

## Week 2: Understanding Results and Finding Alternatives

### June 21-22, 2022 (Day 4 - 5)
* Issue: We have low True Positive rate in our models. 
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
* Issue: Bowman's suggestion: Let's find a potentially better decision boundary. 
    - [Support Vector Machine — Simply Explained](https://towardsdatascience.com/support-vector-machine-simply-explained-fee28eba5496)
    - [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)
        - The advantages of support vector machines are:
            1. Effective in high dimensional spaces.
            2. Still effective in cases where number of dimensions is greater than the number of samples.
            3. Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
            4. Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.
    - [In-Depth: Support Vector Machines](https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html)
    - SVM does not support unsupervised learning. Alternatives: SV Clustering?
* Let's reform our question. I realize we have been addressing the wrong questions for the past couple of days... 
    - While Dr. Bowman's suggestion to use chi = 3 as a separator for our labels is plausible, we should not build a decision boundary based on a scatter using a feature (ustat) and the target (chi). The target value should not be in the scatter. 
    - First, we should use an unsupervised learning approach to find proper labels. Or we can keep using chi = 3 and separate the labels into small chi and large chi. 
    - Second, we should find ways to better feature engineer and feature select our data. Maybe remove some data. This is something to consider since we potentially have an imbalance of labeled data. 

### Week 2 Meeting: 


## Week 3: Improving Models and Feature Engineering

### June 28, 2022 
* Created new feature for dataset based on the absolute difference between 


### June 29, 2022 
https://www.kaggle.com/code/rafjaa/dealing-with-very-small-datasets/notebook 
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html#sklearn.naive_bayes.CategoricalNB 
https://argoshare.is.ed.ac.uk/healthyr_book/should-i-convert-a-continuous-variable-to-a-categorical-variable.html 
https://towardsdatascience.com/8-simple-techniques-to-prevent-overfitting-4d443da2ef7d?gi=8501d866cf11

## Week 4: Improving Models and Feature Engineering

week04

https://python-course.eu/machine-learning/artificial-datasets-with-scikit-learn.php