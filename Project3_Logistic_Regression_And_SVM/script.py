import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pickle

def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]       #50000
    n_features = train_data.shape[1]   #715  
  
    bias=np.ones(shape=(train_data.shape[0],1))  # 50000 x 1    
    new_train_data=np.concatenate((bias,train_data),axis=1) # Added bias to input data. 50000 x 716

    theta=np.zeros(shape=(new_train_data.shape[0],1))  # 50000 x 1    
    one_minus_theta=np.zeros(shape=(new_train_data.shape[0],1))   # 50000 x 1 
   
    newWeights=initialWeights.reshape(new_train_data.shape[1],1)
    w_transpose_x=np.dot(new_train_data,newWeights)
    theta=sigmoid(w_transpose_x)    # 50000 X 1   
 
    log_theta=np.log(theta)
    y=np.dot(labeli.transpose(),log_theta)       
 
    one_minus_yn=np.subtract(1.0,labeli).transpose()
    one_minus_theta=np.subtract(1.0,theta)
    ln_one_minus_theta=np.log(one_minus_theta)
    z=np.dot(one_minus_yn,ln_one_minus_theta)    
   
    error_product= y + z
    error=np.sum(error_product)
    error=(-error)/new_train_data.shape[0] 
   
    error_grad = np.zeros(shape=(new_train_data.shape[0], 1))       
    theta_n_minus_yn=np.zeros(shape=(new_train_data.shape[0],1))
    theta_n_minus_yn=np.subtract(theta,labeli)  # 50000 x 1    

    error_grad=np.dot(new_train_data.transpose(),theta_n_minus_yn)   # 716 x 50000 * 50000 x 1    
    error_grad=error_grad/new_train_data.shape[0]

    return error, error_grad.flatten()



def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))
    bias=np.ones(shape=(data.shape[0],1))  #50000 x 1    
    new_train_data=np.concatenate((bias,data),axis=1) # Added bias to input data. 50000 x 716    
    w_transpose_x=np.dot(new_train_data,W)    
    predicted_label=np.zeros(shape=(new_train_data.shape[0],10))    
    predicted_label=sigmoid(w_transpose_x)  # 50000 x 10    
    label = np.argmax(predicted_label, axis = 1)  # select 1 (highest) out of 10 values of labels as final label for the data 
    label = label.reshape(data.shape[0],1)
    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    
    """
    
    '''
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))
    '''
    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    train_data,labeli=args
    initialWeights=params[:].reshape((n_feature+1,n_class))
    train_data_new=np.ones(shape=(train_data.shape[0],1),dtype = np.float64)
    train_data = np.concatenate((train_data_new, train_data), axis = 1)
    WTransX=np.dot(train_data,initialWeights)
    expWTransX=np.exp(WTransX)
    sumexpWTransX=np.sum(expWTransX,axis=1)
    sumexpWTransX = sumexpWTransX.reshape(train_data.shape[0],1)
    softmax=expWTransX/sumexpWTransX
    logsm=np.log(softmax)
    ce=labeli*logsm
    ce=ce*(-1)
    ce1=np.sum(ce,axis=0)
    ce2=np.sum(ce1)
    error=ce2
    eg=softmax-labeli
    error_grad=np.dot(eg.transpose(),train_data)
    error_grad=error_grad.transpose()
    error_grad=error_grad/train_data.shape[0]
    
    
    print(error)
    return error, error_grad.flatten()
    



def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    #label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    label = np.zeros((data.shape[0], 1))
    bias=np.ones(shape=(data.shape[0],1))  #50000 x 1
    
    new_train_data=np.concatenate((bias,data),axis=1) # Added bias to input data. 50000 x 716
    
    w_transpose_x=np.dot(new_train_data,W)
    
    predicted_label=np.zeros(shape=(new_train_data.shape[0],10))
    
    predicted_label=sigmoid(w_transpose_x)  # 50000 x 10
    
    label = np.argmax(predicted_label, axis = 1)  # select 1 (highest) out of 10 values of labels as final label for the data 
    label = label.reshape(data.shape[0],1)
    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()


# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))
f1=open('params.pickle','wb')
pickle.dump(W,f1)
f1.close()

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################

#Code for calculating accuracy using linear kernel

train_label = np.squeeze(train_label)
clf = SVC(kernel = 'linear')
clf.fit(train_data, train_label)
print("Done fitting for Linear Kernel")
accuracy_train = clf.score(train_data, train_label)
print("accuracy of train: ",accuracy_train)
accuracy_test = clf.score(test_data, test_label)
print("accuracy of test: ",accuracy_test)
accuracy_validation = clf.score(validation_data, validation_label)
print("accuracy of validation: ",accuracy_validation)
print("\n-------------------------------\n")


#Code for calculating accuracy using rbf kernel with gamma = 1

clf = SVC(gamma = 1)
clf.fit(train_data, train_label)
print("Done fitting for RBF kernel with gamma = 1")
accuracy_train = clf.score(train_data, train_label)
print("accuracy of train: ",accuracy_train)
accuracy_test = clf.score(test_data, test_label)
print("accuracy of test: ",accuracy_test)
accuracy_validation = clf.score(validation_data, validation_label)
print("accuracy of validation: ",accuracy_validation)
print("\n-------------------------------\n")

#Code for calculating accuracy using rbf kernel with other parameters default

clf = SVC(kernel = 'rbf')
clf.fit(train_data, train_label)
print("Done fitting for RBF kernel")
accuracy_train = clf.score(train_data, train_label)
print("accuracy of train: ",accuracy_train)
accuracy_test = clf.score(test_data, test_label)
print("accuracy of test: ",accuracy_test)
accuracy_validation = clf.score(validation_data, validation_label)
print("accuracy of validation: ",accuracy_validation)
print("\n-------------------------------\n")

#Code for plotting the graph of accuracy with respect to varying C values

vector = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] #C values stored in a vector
vectorTrain = []
vectorTest = []
vectorValidate = []

for i in vector:
    train_label = np.squeeze(train_label)
    clf = SVC(C = i)
    print("Doing for C: ",i)
    clf.fit(train_data, train_label)
    print("Done fitting")
    accuracy_train = clf.score(train_data, train_label)
    print("accuracy of train: ",accuracy_train)
    accuracy_test = clf.score(test_data, test_label)
    print("accuracy of test: ",accuracy_test)
    accuracy_validation = clf.score(validation_data, validation_label)
    print("accuracy of validation: ",accuracy_validation)
    vectorTrain.append(accuracy_train)
    vectorTest.append(accuracy_test)
    vectorValidate.append(accuracy_validation)
    print("\n-------------------------------\n")


vectorTrain_new = [i * 100 for i in vectorTrain]  #Converted accuracy into percentage
vectorTest_new = [i * 100 for i in vectorTest]
vectorValidate_new = [i * 100 for i in vectorValidate]

accuracyMatrix = np.column_stack((vectorTrain_new, vectorTest_new, vectorValidate_new))
    
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(vector,accuracyMatrix)
plt.title('Accuracy with varying values of C')
plt.legend(('Testing data','Training data', 'Validation data'), loc = 'best') 
plt.xlabel('C values')
plt.ylabel('Accuracy in %') 
plt.show()

"""
Script for Extra Credit Part
"""

# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

f2=open('params_bonus.pickle','wb')
pickle.dump(W_b,f2)
f2.close()

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')