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
* The new dataset can be found [here](../data/preprocessed/week01.csv).

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
* Presented [PowerPoint](../logs/files/Week%2001%20and%2002%20Report.pdf) for Week 1 and Week 2 results.
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
* The new dataset can be found [here](../data/preprocessed/week03_nobin.csv).
* Followed the same procedures as [Week01 - Models.ipynb](../nb/Week01/Week01%20-%20Models.ipynb) in processing and assessing data, except this time we relied only on Logistic Regression as our go-to algorithm. 
* The results are pretty good with high training accuracy of 0.95 and testing accuracy of 0.92. The area under the curve score is high with a score of 0.95. The confusion matrix shows that the model does attempt to identify both groups, with a good True Negative (small chi) and True Positive (large chi).

### June 29, 2022 (Day 8)  
* Adding a new feature may potentially help in improving model accuracy by providing more useful information and speeding up data transformation.
* Created a new dataset with 'diff' as a new feature. Diff is the absolute difference between alph1 and alph2. Diff remains a feature unless stated otherwise. Thus, there is now a total of five features. 
* The new dataset can be found [here](../data/preprocessed/week03_nobin_diff.csv).
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
* The new datasets can be found [here](../data/preprocessed/week04_large_chi_art_data.csv) and [here](../data/preprocessed/week04_overlap_data.csv).
* Confusion matrix shows 174 instances were correctly identified as small chi (True Negative) and 9 were incorrectly identified as 'large chi (False Positive). As expected, the model does not predict perfectly. 

### July 1, 2022 (Day 10)  
* Bowman suggested: 
    - Given the ustat v chi scatter, can you cut from the small and large chi datasets just the data with u* between 5 and 12?  Then we can eyeball the these datasets and see if we spot anything that might distinguish the two.
* Goal: Process only data instances that lie between 5.26 and 12.6. The minimum ustat value for large chi is 5.26 and the maximum ustat value for small chi is 12.6. 
* There are a total of 128 data instances: 98 small chi and 30 large chi.
* The new dataset can be found [here](../data/preprocessed/week03_overlap_data.csv).
* Same process and model assessment as before. There is good testing accuracy and AUC curve. Model predicts small and large chi with errors. Issues may stem from not having enough testing data for large chi.

## Week 4: Improving Models and Feature Engineering

* Results and notes for the week can be found on [Week04 - Restructuring Features.ipynb](../nb/Week04/Week04%20-%20Restructuring%20Features.ipynb) and [Week04 - Generating Artificial Data, Improving Models.ipynb](../nb/Week04/Week04%20-%20Generating%20Artificial%20Data%2C%20Improving%20Models.ipynb).
* The new datasets can be found [here](../data/preprocessed/week03_train_df_sugg01.csv) and [here](../data/preprocessed/week03_test_df_sugg01.csv).

### July 5, 2022 (Day 11)  
* Presented [PowerPoint](../logs/files/Week%2004%20Report.pdf) for Week 4 results.
* Discussed results and future direction.
    - Since data is expensive, it is plausible to create artificial data to use for testing. The data just has to be reasonable and 'predictable' (i.e. we are sure of the classification). 
    - Retry processing and assessment but remove alph1 and alph2 as features.
    - Examining the 'overlap' data further, we want to be able to classify this region. Does not have to be small chi or large chi. 

### July 7, 2022 (Day 13)  
* Explored the overlap data for patterns and relationships among the features (and target).
* There was not a lot of data to work with, since we reduced the already small dataset.
* After observing the heatmap, we see high correlation between the variables alph2 and diff. This was confirmed through the scatter plot of the two features. However, this does not give a lot of useful information because alph2 is part of diff ( = abs difference between alph1 and alph2).
* None of the other scatters displayed meaningful information, even with the addition of the chi labels.

### July 8, 2022 (Day 14)  
* Explored the [Random Module](https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html) of numpy. numpy.random.normal draws random samples from a normal distribution. 
* Decided the best way to create artificial data would be to add noise to the existing data. Noise was created using the numpy.random.normal function and with the same dimension as the original dataset. 
* Plotting the artificial data, we see that the new data resembles the existing dataset. 
* I wanted to use the artificial data for two purposes: 
    1. Use as testing data to see how models performs on more unseen data.
    2. Use as training data to regularize the model. 


### July 9, 2022 (Day 15)  
* Experiment 1: Used artificial data as extra testing data. Tested on LR model from last week with perfect testing and AUC score.
    - Testing data consisted of 114 entries: 38 noisy large chi and  76 randomly selected noisy small chi
    - Achieved perfect AUC score.
    - Thoughts: I was hoping the additional testing data results would prove the model was overfitting. However, it was not able to do so. An imbalance of labeled data and the small size of the data are still problems we need to address. 
* Experiment 2: Used artificial data in the training process. Trained and tested new LR model.  
    - Training data consisted of 290 entries. Training data had 270 entries after train_test_split (67:33 ratio). 10 random noisy large chi and 10 random noisy small chi were added afterwards.
    - Achieved perfect testing and AUC score.
    Thoughts: Similar to the first experiment, I wanted to eliminate the possibility of overfitting. But, I no longer think overfitting is a problem. Bad labels will create problems for any classification problem. 


## Week 5 - 6: Changes

### July 11, 2022 (Day 16)  
* Presented to and discussed results with Chen.
* Chen agreed the labeling of the original dataset with 'small chi' and 'large chi' is potentially problematic, since there is an uneven distribution between the two labels. Suggested unsupervised learning as a way to discover new labels.
* Talked about previous experiences and results with unsupervised learning methods on the dataset. 
* We both agreed three dimensional clustering would be a good place to start to find new labels.  

### July 12 and 13, 2022 (Day 17-18)  
* Presented [PowerPoint](../logs/files/Week%2004%20Report.pdf) for Week 4 results.
* Dr. Bowman noticed something unusual about the original dataset. Hosted another meeting with the rest of the group to talk about it. 

### July 19, 2022 (Day 19)  
* Meeting with Dr. Bowman and Dr. Houston. 
* Presented with updated dataset, which has been purged of duplicate entries (i.e., there are no 1D reactions, only 3D). 
* There is another dataset, containing ONLY the duplicate entries, in which there are both 1D and 3D entries.
* The new purpose for using these datasets is to  take the 1D reactions, figure out how to perform Gaussian Process Regressin on them, so we can predict/modify and achieve 3D results. 
* Current goal is to analyze the purged dataset, which has already been prelabeled 'small chi' and 'large chi.'
* The purged dataset can be found [here](xxx) along with its [small chi](xxx) and [large chi](xxx) components. The duplicates dataset can be found [here](xxx).

### July 20, 2022 (Day 20)
* Performed data exploration on updated dataset. 

### July 21, 2022 (Day 21)
* Transformed the data.
* Looked into preprocessing techniques. Decided to pursue [PowerTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html) from Sci-kit Learn. 
* Implemented Yeo-Johnson Transform on data. 
* Plotted the data on 3D plot.

#### Notes:
* What is power transform?
    - Power transforms refer to a class of techniques that use a power function (like a logarithm or exponent) to make the probability distribution of a variable Gaussian or more-Gaussian like. This is often described as removing a skew in the distribution, although more generally is described as  stabilizing the variance of the distribution.
    - The idea is to apply a transformation to each feature of our dataset. 
*  Why use power transform?
    - The idea is to increase the symmetry of the distribution of the features. If a features is asymmetric, applying a power transformation will make it more symmetric. 
    - Some models may not work properly if the features are not symmetric. For example, models based on distances like KNN or K-means may fail if the distributions are skewed. In order to make these models work, power transformations will symmetrize the features without affecting their predictive power too much.
    - The transformed training dataset can then be fed to a machine learning model to learn a predictive modeling task.
* Box-Cox Transform
    - It is a power transform that assumes the values of the input variable to which it is applied are strictly positive. That means 0 and negative values are not supported.
* Yeo-Johnson Transform
    - Unlike the Box-Cox transform, it does not require the values for each input variable to be strictly positive. It supports zero values and negative values. This means we can apply it to our dataset without scaling it first.
* According to my experience, it’s worth using power transformations when we use models based on distances like KNN, K-means, DBSCAN. 
* Resources:
    - [sklearn.preprocessing.PowerTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html)
    - [When and how to use power transform in machine learning](https://www.yourdatateacher.com/2021/04/21/when-and-how-to-use-power-transform-in-machine-learning/)
    - [How to Use Power Transforms for Machine Learning](https://machinelearningmastery.com/power-transforms-with-scikit-learn/)

### July 23, 2022 (Day 22)
* Performed k-Means clustering on 3d scatter features, used elbow method to choose k value. 
* Clusters appeared reasonable. Will need to examine relationship between clusters and chi values before confirming labels.

##### Notes:
* What is k-Means Clustering?
    - K-means clustering is a very famous and powerful unsupervised machine learning algorithm. Clustering is the process of grouping data samples together into clusters based on a certain feature that they share.
    - A K-means clustering algorithm tries to group similar items in the form of clusters. The number of groups is represented by K.
    - Steps:
        1. First, we need to provide the number of clusters, K, that need to be generated by this algorithm.
        2. Next, choose K data points at random and assign each to a cluster. Briefly, categorize the data based on the number of data points.
        3. The cluster centroids will now be computed.
        4. Iterate the steps below until we find the ideal centroid, which is the assigning of data points to clusters that do not vary.
            - The sum of squared distances between data points and centroids would be calculated first.
            - At this point, we need to allocate each data point to the cluster that is closest to the others (centroid).
            - Finally, compute the centroids for the clusters by averaging all of the cluster’s data points.
* How do we choose k?
    - Elbow Method: an empirical method to find out the best value of k. it picks up the range of values and takes the best among them. It calculates the sum of the square of the points and calculates the average distance.
    - When the value of k is 1, the within-cluster sum of the square will be high. As the value of k increases, the within-cluster sum of square value will decrease.
    - Finally, we will plot a graph between k-values and the within-cluster sum of the square to get the k value. we will examine the graph carefully. At some point, our graph will decrease abruptly. That point will be considered as a value of k.
* Resources:
    - [sklearn.cluster.KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
    - [A Simple Explanation of K-Means Clustering](https://www.analyticsvidhya.com/blog/2020/10/a-simple-explanation-of-k-means-clustering/)
    - [Understanding K-means Clustering in Machine Learning(With Examples)](https://www.analyticsvidhya.com/blog/2021/11/understanding-k-means-clustering-in-machine-learningwith-examples/)
    - [How Does k-Means Clustering in Machine Learning Work?](https://towardsdatascience.com/how-does-k-means-clustering-in-machine-learning-work-fdaaaf5acfa0)

### July 24, 2022 (Day 23)
* Presented [PowerPoint](../logs/files/Week%2006%20Report.pdf) for Week 6 results.

### July 25, 2022 (Day 24)
* Examined the clusters. Wanted to answer two questions for each of the scatter-clusters. 
    1. How many data instances are in each clusters? 
    2. What chi values are in each cluster?
* The first question will determine whether there is an imbalance of instances in each cluster. Ideally, we want each cluster to be approximately the same size. The second question will let us know whether we can create separate labels from each cluster. In other words, if a cluster only has a certain chi value range which only appears in that cluster, them it can be associated with a label.
* No clear labels can be extracted from any fo the scatter-cluster plots.
* Decided to try using k = 2, since we originally had two labels. No clear labels on this one, either. 
* One idea: KM03 (k = 2) has about roughly about the same number of instances for both clusters. Cluster 0 appears to occupy a smaller range compared to Cluster 1. If we can exclude that range of chi values from Cluster 1, then we can potentially create labels for the data. Granted, by doing this: (1) there will be less data in the overall dataset and (2) the model accuracy might suffer a little bit.

### July 26, 2022 (Day 25)
* Decided to add feature 'diff' (= abs difference between alph1 and alph2) and drop features 'alph1' and 'alph2.' I thought it would be a good attempt, since we will have exactly three features for scatter/clustering.
* Same as before, no clear labels can be generated from the clusters. 
* The results from the past two days indicate that there are a lot of overlap 

### July 28, 2022 (Day 26)
* Dr. Bowman suggested reading/skimming this journal [Accurate Molecular-Orbital-Based Machine Learning Energies via Unsupervised Clustering of Chemical Space](https://pubs.acs.org/doi/10.1021/acs.jctc.2c00396).
    - The group uses Gaussian mixture model (GMM) for determining clusters automatically, without the need for user-specified paramters and training of an additional classifier. 
* Did more reading on GMM.

* Resources: 
    * [sklearn.mixture.GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
    * [2.1. Gaussian mixture models](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
    * [Gaussian Mixture Model](https://www.geeksforgeeks.org/gaussian-mixture-model/)
    * [Gaussian Mixture Models Explained](https://towardsdatascience.com/gaussian-mixture-models-explained-6986aaf5a95)

#### Notes:
* A Gaussian mixture model is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. 
    - One can think of mixture models as generalizing k-means clustering to incorporate information about the covariance structure of the data as well as the centers of the latent Gaussians.


* One important characteristic of K-means is that it is a hard clustering method, which means that it will associate each point to one and only one cluster.
    - A limitation to this approach is that there is no uncertainty measure or probability that tells us how much a data point is associated with a specific cluster.
* A Gaussian Mixture is a function that is comprised of several Gaussians, each identified by k ∈ {1,…, K}, where K is the number of clusters of our dataset. Each Gaussian k in the mixture is comprised of the following parameters:
    - A mean μ that defines its centre.
    - A covariance Σ that defines its width. This would be equivalent to the dimensions of an ellipsoid in a multivariate scenario.
    - A mixing probability π that defines how big or small the Gaussian function will be.

### July 29, 2022 (Day xx)
* 