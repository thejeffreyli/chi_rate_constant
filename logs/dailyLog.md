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
* I wanted to see whether there was an alternate (and possible more systematic) way of finding this decision boundary. Hence, I used [DBSCAN from Scikit Learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html). Density-based spatial clustering of applications with noise or DBSCAN is a density-based clustering non-parametric algorithm: given a set of points in some space, it groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions (whose nearest neighbors are too far away). More info on this algorithm can be found on the Sciki Learn site and [Wikipedia](https://en.wikipedia.org/wiki/DBSCAN).


