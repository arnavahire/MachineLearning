import numpy as np
from scipy.optimize import minimize
from numpy.linalg import inv
from math import sqrt
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD 
    unique_y=np.unique(y)  # 5*0 1d array
    
    means=np.zeros(shape=(X.shape[1],unique_y.shape[0])) # 2*5
    
    for i in range(unique_y.shape[0]):
        y_index=np.where(y==unique_y[i])
        indices=y_index[np.all(y_index == 0, axis=0)]    # Choose those indices of x where y=1/2/3/4/5
        X_1=X[indices]                                   # new X vector corresponding to each y
        mu=X_1.mean(0)     # 2*0 matrix i.e one d array
        means[0][i]=mu[0]  # dimension 1 mean
        means[1][i]=mu[1]  # dimension 2 mean
 
    covmat=np.cov(means)   # single 2*2 covariance matrix 
    
    return means,covmat

#----------------------------------------------------------------------------------------------------------

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    unique_y=np.unique(y)  # 5*0 1d array
    
    means=np.zeros(shape=(X.shape[1],unique_y.shape[0])) # 2*5
    
    covmats=np.zeros(shape=(unique_y.shape[0],X.shape[1],X.shape[1])) # 5*2*2 since we will store 5 two by two matrices
    
    for i in range(unique_y.shape[0]):
        y_index=np.where(y==unique_y[i])
        indices=y_index[np.all(y_index == 0, axis=0)]   # Choose those indices of x where y=1/2/3/4/5
        X_1=X[indices]                                  # new X vector corresponding to each y
        mu=X_1.mean(0) # 2*0 matrix i.e one d array        
        means[0][i]=mu[0]  # dimension 1 mean
        means[1][i]=mu[1]  # dimension 2 mean
        
        # Get covariance matrix    ((x-mu)T*(x-mu))/N       N= total input x's for each class y    
        
        Xi_minus_mu1=np.subtract(X_1,mu)    # Get covariance matrix for y==1,2,3,4,5
        Xi_minus_mu1_trans=Xi_minus_mu1.transpose()
        ans=np.dot(Xi_minus_mu1_trans,Xi_minus_mu1)
        cov_matrix_1=ans/Xi_minus_mu1.shape[0]                 
        covmats[i]=cov_matrix_1   # final covariance matrix for 'each class'

    return means,covmats

#-----------------------------------------------------------------------------------------------------------    

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    x_prediction_given_y=np.zeros(shape=(Xtest.shape[0],means.shape[1]))   #100*5
 
    count=-1
    for x in Xtest:
        count=count+1    # for accessing the xth row of x_prediction_given_y
        for i in range(x_prediction_given_y.shape[1]): 
            x_min=np.subtract(x,means[:,i])             # 2*0 since its a 1d array  (x-mu)
            x_min = x_min[:, np.newaxis]                # increase dimension by 1 so that x_min becomes 2*1 matrix
            
            #Calculating numerator
            x_min_trans=x_min.transpose()     # (x-mu)T 1*2
            epsilon_inv=inv(covmat)       # inv(covariance_matrix) 2*2            
            a=np.dot(epsilon_inv,x_min)       # a= 2*1 vector           
            exponent=np.dot(x_min_trans,a)    # exponent=1*1 vector           
            numerator=np.exp(-0.5*exponent)            

            #Calculating denominator
            two_pie=2*(22/7)
            
            det_cov_matrix=np.linalg.det(covmat)
            root_det_cov_matrix=sqrt(det_cov_matrix)
            denominator=pow(two_pie,x_prediction_given_y.shape[1]/2)*root_det_cov_matrix
            x_prediction_given_y[count][i]=numerator/denominator
    
    # Calculate posterior
    
    y_prediction=np.zeros(shape=(Xtest.shape[0],means.shape[1]))   #100*5
    count = -1
    for x in Xtest:
        count=count+1    # for accessing the xth row of x_prediction_given_y
        for i in range(y_prediction.shape[1]):  
            numerator= x_prediction_given_y[count][i]
            denominator= x_prediction_given_y[count][0]+ x_prediction_given_y[count][1]+ x_prediction_given_y[count][2]+ x_prediction_given_y[count][3]+ x_prediction_given_y[count][4]
            y_prediction[count][i]=numerator/denominator

    # Selecting the index(class) with highest value as our final class
    ypred = np.argmax(y_prediction, axis = 1)
    ypred=ypred+1        # ypred contains values 0 to 4. but are final labels are 1 to 5
    ytest=ytest.flatten()

    count=0
    for i in range(ypred.shape[0]):
        if(ypred[i]==ytest[i]):      # if true label equals predicted label increase count
            count=count+1
    acc = count/ypred.shape[0]         # accuracy
    
    return acc,ypred

#----------------------------------------------------------------------------------------

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    x_prediction_given_y=np.zeros(shape=(Xtest.shape[0],means.shape[1]))   #100*5
 
    count=-1
    for x in Xtest:
        count=count+1    # for accessing the xth row of x_prediction_given_y
        for i in range(x_prediction_given_y.shape[1]):
            x_min=np.subtract(x,means[:,i])             # 2*0 since its a 1d array  (x-mu)
            x_min = x_min[:, np.newaxis]   # increase dimension by 1 so that it becomes 2*1
            
            #Calculating numerator
            x_min_trans=x_min.transpose()     # (x-mu)T 1*2
            epsilon_inv=inv(covmats[i])       # inv(covariance_matrix) 2*2
            a=np.dot(epsilon_inv,x_min)       # a=2*1 vector
            exponent=np.dot(x_min_trans,a)    # exponent=1*1 vector
            numerator=np.exp(-0.5*exponent)
            
            #Calculating denominator
            two_pie=2*(22/7)
            
            det_cov_matrix=np.linalg.det(covmats[i])
            root_det_cov_matrix=sqrt(det_cov_matrix)
            denominator=pow(two_pie,x_prediction_given_y.shape[1]/2)*root_det_cov_matrix
            x_prediction_given_y[count][i]=numerator/denominator
    
    # Calculate posterior
    y_prediction=np.zeros(shape=(Xtest.shape[0],means.shape[1]))   #100*5 
    count = -1
    for x in Xtest:
        count=count+1    # for accessing the xth row of x_prediction_given_y
        for i in range(y_prediction.shape[1]):  
            numerator= x_prediction_given_y[count][i]
            denominator= x_prediction_given_y[count][0]+ x_prediction_given_y[count][1]+ x_prediction_given_y[count][2]+ x_prediction_given_y[count][3]+ x_prediction_given_y[count][4]
            y_prediction[count][i]=numerator/denominator
            
    # Selecting the index(class) with highest value as our final class
    ypred = np.argmax(y_prediction, axis = 1)
    ypred=ypred+1     # ypred contains values 0 to 4. but are final labels are 1 to 5
    ytest=ytest.flatten()

    count = 0    
    for i in range(ypred.shape[0]):
        if(ypred[i]==ytest[i]):     # if true label equals predicted label increase count
            count=count+1
    acc = count/ypred.shape[0]        # accuracy
        
    return acc,ypred

#---------------------------------------------------------------------------------------------------

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
    # IMPLEMENT THIS METHOD
    X_trans=X.transpose()
    X_trans_X=np.dot(X_trans,X)
    X_trans_X_inv=inv(X_trans_X) 
    X_trans_y=np.dot(X.transpose(),y)
    w=np.dot(X_trans_X_inv,X_trans_y)                                                   
    return w

#-------------------------------------------------------------------------------------------

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD                                                   
    i_matrix = np.identity(X.shape[1])
    lambd_i = np.dot(lambd,i_matrix)
    X_trans=X.transpose()
    X_trans_X=np.dot(X_trans,X)
    exp_1 = np.add(lambd_i,X_trans_X)
    exp_1 = inv(exp_1)
    X_trans_y=np.dot(X_trans,y)
    w = np.dot(exp_1, X_trans_y)
    return w

#-----------------------------------------------------------------------------------------

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    # IMPLEMENT THIS METHOD
    X_w=np.dot(Xtest,w)
    y_minus_Xw=np.subtract(ytest,X_w)
    y_minus_Xw_trans=y_minus_Xw.transpose()
    product=np.dot(y_minus_Xw_trans,y_minus_Xw)
    mse=product/Xtest.shape[0]
    return mse
 
#--------------------------------------------------------------------------------------------

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD
    w = w.reshape(len(w),1)
    X_w = np.dot(X,w)
    y_minus_Xw = np.subtract(y, X_w)
    y_minus_Xw_t = np.transpose(y_minus_Xw)
    exp_1 = np.dot(y_minus_Xw_t, y_minus_Xw)
    exp_1 = exp_1/2
    w_t = np.transpose(w)
    exp_2=np.dot(w_t,w)
    exp_2 = lambd*exp_2
    exp_2 = exp_2/2
    error = exp_1 + exp_2
    
    w_transX=np.dot(X,w)
    w_transXminusy= np.subtract(w_transX,y)
    exp1=np.dot(X.transpose(),w_transXminusy)
    exp2=lambd*w
    exp3=exp1+exp2
    error_grad=exp3    
    error_grad=error_grad.flatten()
    
    return error, error_grad

#-------------------------------------------------------------------------------------------------

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1)) 
	
    # IMPLEMENT THIS METHOD
    Xd = np.zeros(shape=(x.shape[0],(p+1)), dtype = np.float64)
        
    for row in range(x.shape[0]):
        for var in range(p+1):
            Xd[row][var] = x[row]**var
     
    return Xd

#------------------------------------------------------------------------------------------------

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
sldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(sldaacc))


# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()


# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mse = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mse_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept ',str(mse))
print('MSE with intercept ',str(mse_i))


# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))

for lambd in lambdas: #loop will execute 101 times
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)  #Optimal lambda value coming out to be 0.06
    i = i + 1 
    
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()


# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
        
    
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = 0.06 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))

for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y) #No regularization or lambda = 0
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest) #No regularization or lambda = 0
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y) #Regularization lambda = 0.06
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest) #Regularization lambda = 0.06

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()

#--------------------------------------------------------------------------------------------------