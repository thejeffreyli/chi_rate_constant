## Week 0: Reading Literature

### Original Work by Dr. Paul Houston, Apurba Nandi, and Dr. Joel Bowman
* [A Machine Learning Approach for Prediction of Rate Constants](https://pubs.acs.org/doi/10.1021/acs.jpclett.9b01810)
* [A Machine Learning Approach for Rate Constants. II. Clustering, Training, and Predictions for the O(3P) + HCl → OH + Cl Reaction](https://pubs.acs.org/doi/10.1021/acs.jpca.0c04348)


## Week 1: Getting Familiar with Data

### June 13, 2022 (Day 1)
Updates:
* Began looking at the chi dataset. The .dat file contains 404 data instances describing parameters and chi. Parameters include ustat, alph1, alph2, and beta. 
* I wanted to first look at the relationship with each of the parameters with Chi. Relationships were plotted on a scatter plot with chi on the y-axis. 
    - ustat vs chi: We see most data instances are heavily concentrated on the lower left region. Remaining data intances are scattered across the right, appearing in an upward manner. 
    - alph1 vs chi: There does not appear to be any noticeable relationship between both variables. Appears evenly distributed. 
    - alph1 vs chi: More data points are near alph2 = 0. 
    - beta vs chi: There appears to be a gaussian spread, with more data points clustered around beta = 50. However, it is important to note the clustering of poitns at beta = 10 and beta = 90. 
* Since we want to develop a classification model for this data, I also wanted to evaluate the spread of data so I can make categorical data. This was done using a box plot.
* Houston and group tried to develop a classification model using two labels: small chi and large chi. The decision boundary for their work was found by eyeing the data; the decision boundary noted is at chi = 3. When chi is less than 3, the data instance is labeled as "small chi." When chi is greater than or equal to 3, the data instance is labeled as "large chi." 
* I wanted to see whether there was an alternate (and possible more systematic) way of finding this decision boundary. Hence, I used [DBSCAN from Scikit Learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html). Density-based spatial clustering of applications with noise or DBSCAN is a density-based clustering non-parametric algorithm: given a set of points in some space, it groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions (whose nearest neighbors are too far away). More info on this algorithm can be found on the Scikit Learn site and [Wikipedia](https://en.wikipedia.org/wiki/DBSCAN).
* Setting the epsilon value to 1 (local radius) and the number of samples to six, we achieve a scatterplot that closely resembles having a boundary line at chi = 3. However, it is important to note that while blue represents one cluster, red represents outliers when we are using DBSCAN. 

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


