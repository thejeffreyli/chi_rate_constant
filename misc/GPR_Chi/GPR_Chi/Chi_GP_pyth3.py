import numpy as np
import os
import sys
import time
import random

# We use Sklearn libraries to run GP

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel,\
     ConstantKernel as C,\
     RationalQuadratic as RQ,\
     ExpSineSquared as ESS,\
     DotProduct as DP
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split

'''
'Authors: Apurba Nandi, Joel M. Bowman, Paul Houston'
'Location: Emory University, Department of Chemistry, Atlanta, USA'
'Contact: apurba.nandi@emory.edy, jmbowma@emory.edu;'
'Date: May 8, 2020'
'Version: ML-Rate-Cons-V1.0'
'''

#----------------------------------
# FUNCTIONS
def pred_gp(gp,X):
#   gp --> trained GP model
#   X  --> Points to predict
    y_pred, sigma = gp.predict(np.atleast_2d(X), return_std=True)
    return y_pred, sigma

#----------------------------------------------
#                 MAIN
#----------------------------------------------

start_time = time.time()

print ("***********************************************************")
print ("--------Gaussian Process for Rate Constant Prediction---------")
print ("Authors: Apurba Nandi, Joel M. Bowman, Paul Houston")
print ("Location: Emory University, Department of Chemistry, Atlanta, USA")
print ("Contact: apurba.nandi@emory.edy, jmbowma@emory.edu;")
print ("Date: May 8, 2020")
print ("Version: ML-Rate-Cons-V1.0")
print ("**GP is performed from Sklearn Libraries. We gratly thankful to Rodrigo A. Vargas Hernandez and Roman V. Krems to provide their GP program which was used to fit potential energy surface.**")
print ("***********************************************************")

#------------------------------------------------------------

# File name specification and cluster specification to run GP.
#-------------------------------------------------------------
f_test_data = input("Please Enter the name of the data file for prediction: \n")
cluster_val = int(input("Which Cluster do you want: For Small-Chi cluster enter 1 or For Large-Chi cluster enter 2:  \n"))
#----------------------------------------------------------
# Begining of if statement.
# This part is for Small-Chi cluster training and testing
#----------------------------------------------------------
if cluster_val==1:
	print ("Small-Chi cluster is in use")
	f_train_data = "Small_chi.dat"    # Small-Chi Cluster training data file
	f_train_out= "Small_chi_train.out"      # Output file for the training data
	f_test_out = "Small_chi_predict.out"  # Output file for the test data
	f_theta = "gp_theta.dat"                     # File contains Theta values
	f_alpha = "gp_alpha.dat"                     # File contains Alpha values
	gwd = os.getcwd() # ---> GENERAL WORKING DIRECTORY
	gwd = gwd + '/'
	os.chdir(gwd)
	print ("------------------------------")
	print ("data will be taken from: ", gwd)
	print ("data in file: ", f_train_data)
	print ("GP prediction output in file: ", f_test_out)
	print ("------------------------------")
	#----------------------------------
	# READ DATA
	#  Read the training points from Small-Chi cluster data file
	if os.path.isfile(gwd + f_train_data):
		X = np.loadtxt(gwd + f_train_data, usecols=range(4))
		y = np.loadtxt(gwd + f_train_data, usecols=(4,))
		train_set = np.column_stack((X, y))
		N_training, N_dimension = X.shape
		d = N_dimension
	else:
		print (" file " + f_data + " does not exist!!")
		sys.exit("Error message")
	if os.path.isfile(gwd + f_test_data):
		Z = np.loadtxt(gwd + f_test_data, usecols=range(4))
		test_set = np.column_stack((Z))
		N_test = Z.shape[0]
	else:
		print (" file " + f_test_data + " does not exist!!")
		sys.exit("Error message")
	print ("---------------------------------")
	print ("Training data:")
	print ("Number of dimensions: ", N_dimension )
	print ("Number of training points: ", N_training )
	print ("Test data:")
	print ("Number of test points: ", N_test )
	print ("---------------------------------")
	print ("***********************************************************")
	for i in range(N_test):
		if 0.16 < np.array(Z[i,0]) > 12.56 or 1.03 < np.array(Z[i,1]) > 3.51 or 0.71 < np.array(Z[i,2]) > 95.12 or 11.60 < np.array(Z[i,3]) > 89.30 :
			print ("**Warning: Data is outside of the training data range. Test_Data:", i+1)

		else:
			print ("**Test data is okay for prediction. Test_Data:", i+1)

	print ("***********************************************************")
	print ("---------------------------------")
	ck = C(1.0, (1e-2, 1e2))
	k = ck * RBF(length_scale=(1.0,1.0,1.0,1.0), length_scale_bounds=(1e-3, 1e+3)) # Range of length scale parameter for Small-Chi cluster. You can change this range.
	k = k + ck * WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-6, 1e-4))    # Range of sigma noise parameter for Small-Chi cluster. You can change this range.
	print (k)
	#---------------------------------
	# TRAIN GP MODEL WITH COORDINATES
	print ("Training GP model normal coordinates")
	gp = GaussianProcessRegressor(kernel=k,n_restarts_optimizer=20)
	gp.fit(np.atleast_2d(X), y)
	print("Learned kernel (GP): %s" % gp.kernel_)
	print("Log-marginal-likelihood (GP): %.3f" % gp.log_marginal_likelihood(gp.kernel_.theta))
	np.savetxt(gwd + f_theta, gp.kernel_.theta)
	np.savetxt(gwd + f_alpha, gp.alpha_)
	print ("Prediction with GP model")
	y_pred1, sigma1 = pred_gp(gp, X)
	ygp1 = np.column_stack((y, y_pred1, abs(y-y_pred1), sigma1)) #Printing on output file for Small-Chi training data: Training-chi, predicted-chi, absolute error and uncertainty.
	np.savetxt(gwd + f_train_out, ygp1)
	rmse = np.sqrt(mse(y, y_pred1))
	print ("Training RMSE = ", rmse)
	y_pred2, sigma2 = pred_gp(gp, Z)
	ygp2 = np.column_stack((Z, y_pred2, sigma2))                    #Printing on output file for Small-Chi test data: predicted-chi and uncertainty.
	np.savetxt(gwd + f_test_out, ygp2)
	print ("-------------------------")
	#----------------------------------
#----------------------------------------------------------
# This part is for Large-Chi cluster training and testing
#----------------------------------------------------------
if cluster_val==2:
        print ("Large-Chi cluster is in use")
        f_train_data = "Large_chi.dat"     # Small-Chi Cluster training data file
        f_train_out= "Large_chi_train.out"       # Output file for the training data
        f_test_out = "Large_chi_predict.out"   # Output file for the test data
        f_theta = "gp_theta.dat"                      # File contains Theta values
        f_alpha = "gp_alpha.dat"                      # File contains Alpha values
        gwd = os.getcwd() # ---> GENERAL WORKING DIRECTORY
        gwd = gwd + '/'
        os.chdir(gwd)
        print ("------------------------------")
        print ("data will be taken from: ", gwd)
        print ("data in file: ", f_train_data)
        print ("GP prediction output in file:", f_test_out)
        print ("------------------------------")
        #----------------------------------
        # READ DATA
        #  Read the training points from Large-Chi cluster data file
        if os.path.isfile(gwd + f_train_data):
                X = np.loadtxt(gwd + f_train_data, usecols=range(4))
                y = np.loadtxt(gwd + f_train_data, usecols=(4,))
                train_set = np.column_stack((X, y))
                N_training, N_dimension = X.shape
                d = N_dimension
        else:
                print (" file " + f_data + " does not exist!!")
                sys.exit("Error message")
        if os.path.isfile(gwd + f_test_data):
                Z = np.loadtxt(gwd + f_test_data, usecols=range(4))
                test_set = np.column_stack((Z))
                N_test = Z.shape[0]
        else:
                print (" file " + f_test_data + " does not exist!!")
                sys.exit("Error message")
        print ("---------------------------------")
        print ("Training data:")
        print ("Number of dimensions: ", N_dimension)
        print ("Number of training points: ", N_training)
        print ("Test data:")
        print ("Number of test points: ", N_test)
        print ("---------------------------------")
        print ("***********************************************************")
        for i in range(N_test):
                if 5.27 < np.array(Z[i,0]) > 15.75 or 1.46 < np.array(Z[i,1]) > 3.51 or 1.46 < np.array(Z[i,2]) > 6.24 or 13.60 < np.array(Z[i,3]) > 89.30 :
                        print ("**Warning: Data is outside of the training data range. Test_Data:", i+1)

                else:
                        print ("**Test data is okay for prediction. Test_Data:", i+1)

        print ("***********************************************************")
        print ("---------------------------------")
        ck = C(1.0, (1e-2, 1e2))
        k = ck * RBF(length_scale=(1.0,1.0,1.0,1.0), length_scale_bounds=(1e-1, 1e+3))  # Range of length scale parameter for Small-Chi cluster. You can change this range.
        k = k + ck * WhiteKernel(noise_level=1e-1, noise_level_bounds=(1e-2, 1e-0))     # Range of sigma noise parameter for Small-Chi cluster. You can change this range.
        print (k)
        #--------------------------------
        # TRAIN GP MODEL WITH COORDINATES
        print ("Training GP model normal coordinates")
        gp = GaussianProcessRegressor(kernel=k,n_restarts_optimizer=20)
        gp.fit(np.atleast_2d(X), y)
        print("Learned kernel (GP): %s" % gp.kernel_)
        print("Log-marginal-likelihood (GP): %.3f" % gp.log_marginal_likelihood(gp.kernel_.theta))
        np.savetxt(gwd + f_theta, gp.kernel_.theta)
        np.savetxt(gwd + f_alpha, gp.alpha_)
        print ("Prediction with GP model")
        y_pred1, sigma1 = pred_gp(gp, X)
        ygp1 = np.column_stack((y, y_pred1, abs(y-y_pred1), sigma1))    #Printing on output file for Large-Chi training data: Training-chi, predicted-chi, absolute error and uncertainty.
        np.savetxt(gwd + f_train_out, ygp1)
        rmse = np.sqrt(mse(y, y_pred1))
        print ("Training RMSE = ", rmse)
        y_pred2, sigma2 = pred_gp(gp, Z)
        ygp2 = np.column_stack((Z,y_pred2, sigma2))                      #Printing on output file for Large-Chi test data: predicted-chi and uncertainty.
        np.savetxt(gwd + f_test_out, ygp2)
        #rmse = np.sqrt(mse(w, y_pred2))
        print ("-------------------------")
        #----------------------------------


print("--- %s seconds ---" % (time.time() - start_time))
