# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 22:39:06 2016

@author: mamid
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 19:10:41 2016

@author: mamid
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as math
import pylab as pl

def compute_error_for_graph_given_points(thetha1,x,y):
    totalError = 0
    max_x=np.amax(x)
    min_x=np.amin(x)
    mean_x=np.mean(x)
    for i in range(1, len(x)):
      feature_x= (x[i]-mean_x)/(max_x-min_x)
      h_thetha= 1/(1+np.e**(-x[i]*thetha1))
      #np.exp(thetha1*feature_x)/(1+np.exp(thetha1*feature_x))
      #print("error here is....",h_thetha)
      totalError += (((y[i])*(math.log(h_thetha)))+((1-y[i])*(math.log(1-h_thetha))))
      #print("total error here is....",totalError)
    error_features=-(totalError / (float(len(x))))
    #print("total error after summation here is....",error_features)
      
    return error_features

def step_gradient(thetha_current, x,y, learningRate):
    thetha_gradient = 0
    N = float(len(x))
    max_x=np.amax(x)
    min_x=np.amin(x)
    mean_x=np.mean(x)
    for i in range(1, len(x)):   
        feature_x= (x[i]-mean_x)/(max_x-min_x)
        h_thetha= 1/(1+np.e**(-x[i]*thetha_current))
        #print("hypothesis is::::",h_thetha)
        #np.exp(thetha_current*feature_x)/(1+np.exp(thetha_current*feature_x))
        exp_factor=np.e**(-thetha_current*x[i])
        x_factor=((x[i]*exp_factor)/((1+exp_factor)*(1+exp_factor)))
        #print("x factor is::::",x_factor)
        thetha_gradient +=  (((y[i])*(math.log(h_thetha)))+((1-y[i])*(math.log(1-h_thetha))))*x_factor
       
    new_thetha = thetha_current - ((learningRate/N) * thetha_gradient)
   
    #print("thetha after gradient descent",new_thetha)
    return new_thetha
    
def gradient_descent_runner(x,y, starting_thetha, learning_rate, num_iterations):
    thetha1 = starting_thetha
    for i in range(num_iterations):
        new_thetha = step_gradient(thetha1, x,y, learning_rate)    
    
    return new_thetha
    
    
def sigmoid(b,x):
    a = []
    max_x=np.amax(x)
    min_x=np.amin(x)
    mean_x=np.mean(x)
   # print("length of x is:::",len(x))
   # print("value of b is::::",b)
    for item in range(len(x)):  
        #print("item is::::",item)
        feature_x = (item-mean_x)/(max_x-min_x)
        #print("value of feature x is:",feature_x)
    #y=np.exp(b*feature_x)/(1+np.exp(b*feature_x))
        a.append(1/(1+np.e**(-(x[item]*b))))
    print ("value of a is:::",len(a))
    return a
    
    
def run():
    points_training = pd.read_csv("training.csv", delimiter=",")
    points_test = pd.read_csv("test.csv", delimiter=",")
    x = np.array(points_training['Texture'])
    y = np.array(points_training['Recurrent'])
    x_test = np.array(points_test['Texture'])
    y_test = np.array(points_test['Recurrent'])
    print("test values of x are:::",len(x_test))
    #y_actual = pd.Series(points_training['Recurrent'],name='Actual')
    #y_predicted=pd.Series(points_test['Recurrent'],name='Predicted')
    #print("actual y is:::",y_actual)
    #print("predicted y is:::",y_predicted)
    
    x_features= []
    y_predicted = []
    #print (x,y)
    learning_rate = 0.01
    initial_b = -0.1
    num_iterations = 1000
    threshold= 0.12
   # plt.show()
    print ("Starting gradient descent at b = {0}, error = {1}".format(initial_b, compute_error_for_graph_given_points(initial_b, x,y)))
    print ("Running...")  
    b = gradient_descent_runner(x,y, initial_b, learning_rate, num_iterations)    
    max_x=np.amax(x_test)
    min_x=np.amin(x_test)
    mean_x=np.mean(x_test)
    for item in range(len(x_test)):
        feature_x= (item-mean_x)/(max_x-min_x)
        x_features.append(x_test[item])
    
    #y=np.exp(b*feature_x)/(1+np.exp(b*feature_x))
    #print("value of feature x is:::",feature_x)
    #x_features.append(feature_x)
    #print("value of feature x after appending is:::",x_features)
    #x_array = np.array(x_features)
   # print("value of feature x after appending array is:::",x_array)
    plt.scatter(x_features, y_test)
    y_sigmoid=sigmoid(b,x_test)
    print("sigmoid y values are:::",y_sigmoid)
    for item in y_sigmoid:
        if item < threshold :
           y_predicted.append(0)
        else:
           y_predicted.append(1)
    y_actual = pd.Series(y_test,name='Actual')
    y_predict=pd.Series(y_predicted,name='Predicted')
    print("predicted y len values are:::",len(y_predict))
    #print("actual values of y are::::",y_actual)
    
    df_confusion = pd.crosstab(y_actual, y_predict,rownames=['Actual'], colnames=['Predicted'], margins=True)
    true_neg = df_confusion[0][0]
    false_pos = df_confusion[1][0]
    true_pos = df_confusion[1][1]
    false_neg = df_confusion[0][1]
    print("confusion matrix is::",df_confusion)
    print("true neg is:::",df_confusion[0][0])
    print("false positive is:::",df_confusion[0][1])
    print("false neg is:::",df_confusion[1][0])
    print("true positive is:::",df_confusion[1][1])
    print("true neg is:::",true_neg)
    print("false positive is:::",false_pos)
    print("false neg is:::",false_neg)
    print("true positive is:::",true_pos)
    precision = true_pos/(true_pos+false_pos)
    recall = true_pos/(true_pos+false_neg)
    f1_score= 2*((precision*recall)/(precision+recall))
    print("precsion is::::",precision)
    print("Recall is::::",recall)
    print("f1-score is::::",f1_score)
    
    plt.ylabel('Recurrent')
    plt.xlabel('Texture')
    plt.plot(x_features,y_predict, color='r')
    plt.show()
    #print ('F1 score:', f1_score(y_actual, y_predict))
    #print("x value is:::",x)
    #print("y value is::::",y)
    labels= ['texture','recurrent']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(df_confusion)
    pl.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    pl.xlabel('Predicted')
    pl.ylabel('True')
    pl.show()


# This is the ROC curve
    #plt.plot(false_pos,true_pos)
    #plt.show() 

# This is the AUC
   # auc = np.trapz(true_pos,false_pos)
    
    print("After {0} iterations b = {1},error = {2}".format(num_iterations, b, compute_error_for_graph_given_points(b, x,y)))

    
   
if __name__ == '__main__':
    run()

