
"""
Created on Sun Sep 18 19:10:41 2016

@author: mamid
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as math

def compute_error_for_graph_given_points(thetha1,thetha2,x1,x2,y):
    totalError = 0
    max_x1=np.amax(x1)
    min_x1=np.amin(x1)
    mean_x1=np.mean(x1)
    max_x2=np.amax(x2)
    min_x2=np.amin(x2)
    mean_x2=np.mean(x2)
    for i in range(1, len(x1)):
      feature_x1= (x1[i]-mean_x1)/(max_x1-min_x1)
      feature_x2= (x2[i]-mean_x2)/(max_x2-min_x2)
      h_thetha= 1/(1+np.e**(-((feature_x1*thetha1)+(feature_x2*thetha2))))
      #np.exp(thetha1*feature_x)/(1+np.exp(thetha1*feature_x))
      #print("error here is....",h_thetha)
      totalError += (((y[i])*(math.log(h_thetha)))+((1-y[i])*(math.log(1-h_thetha))))
     # print("total error here is....",totalError)
    error_features=-(totalError / (float(len(x1))))
    #print("total error after summation here is....",error_features)
      
    return error_features

def step_gradient(thetha_current1,thetha_current2, x1,x2,y, learningRate,l_lambda):
    thetha_gradient1 = 0
    thetha_gradient2 = 0
    N = float(len(x1))
    max_x1=np.amax(x1)
    min_x1=np.amin(x1)
    mean_x1=np.mean(x1)
    max_x2=np.amax(x2)
    min_x2=np.amin(x2)
    mean_x2=np.mean(x2)
    for i in range(1, len(x1)):   
        feature_x1= (x1[i]-mean_x1)/(max_x1-min_x1)
        feature_x2= (x2[i]-mean_x2)/(max_x2-min_x2)
        h_thetha= 1/(1+np.e**(-((feature_x1*thetha_current1)+(feature_x2*thetha_current1))))
       # print("hypothesis is::::",h_thetha)
        #np.exp(thetha_current*feature_x)/(1+np.exp(thetha_current*feature_x))
        exp_factor=np.e**(-((thetha_current1*feature_x1)+(thetha_current2*feature_x2)))
        x1_factor=((x1[i]*exp_factor)/((1+exp_factor)*(1+exp_factor)))
        x2_factor=((x2[i]*exp_factor)/((1+exp_factor)*(1+exp_factor)))
        #print("x factor is::::",x1_factor)
        thetha_gradient1 +=  (((y[i])*(math.log(h_thetha)))+((1-y[i])*(math.log(1-h_thetha))))*x1_factor
        thetha_gradient2 +=  (((y[i])*(math.log(h_thetha)))+((1-y[i])*(math.log(1-h_thetha))))*x2_factor
       
    new_thetha1 = thetha_current1*(1-(learningRate*l_lambda/N)) - ((learningRate/N) * thetha_gradient1)
    new_thetha2 = thetha_current2*(1-(learningRate*l_lambda/N)) - ((learningRate/N) * thetha_gradient2)
   
    #print("thetha after gradient descent",new_thetha1)
    return  [new_thetha1,new_thetha2]
    
def gradient_descent_runner(x1,x2,y, starting_thetha1,starting_thetha2, learning_rate,l_lambda, num_iterations):
    thetha1 = starting_thetha1
    thetha2 = starting_thetha2
    for i in range(num_iterations):
        [new_thetha1,new_thetha2] = step_gradient(thetha1,thetha2, x1,x2,y, learning_rate,l_lambda)    
    #print("",)
    return [new_thetha1,new_thetha2]
    
    
def sigmoid(thetha1,thetha2,x1,x2):
    a = []
    max_x1=np.amax(x1)
    min_x1=np.amin(x1)
    mean_x1=np.mean(x1)
    max_x2=np.amax(x2)
    min_x2=np.amin(x2)
    mean_x2=np.mean(x2)
    #print("length of x is:::",len(x))
    #print("value of b is::::",b)
    for i in range(len(x1)):  
       # print("item is::::",item)
        feature_x1= (x1[i]-mean_x1)/(max_x1-min_x1)
        feature_x2= (x2[i]-mean_x2)/(max_x2-min_x2)
        #print("value of feature x is:",feature_x)
    #y=np.exp(b*feature_x)/(1+np.exp(b*feature_x))
        a.append(1/(1+np.e**(-((feature_x1*thetha1)+(feature_x2*thetha2)))))
    #print ("value of a is:::",a)
    return a
    
    
def run():
    points_training = pd.read_csv("training.csv", delimiter=",")
    points_test = pd.read_csv("test.csv", delimiter=",")
    x1 = np.array(points_training['Texture'])
    x2 = np.array(points_training['Area'])
    y = np.array(points_training['Recurrent'])
    x1_test = np.array(points_test['Texture'])
    x2_test = np.array(points_test['Area'])
    y_test = np.array(points_test['Recurrent'])
    x_features= []
    y_predicted = []
    #print (x,y)
    learning_rate = 0.001
    initial_thetha1 = 0
    initial_thetha2 = -0.01
    num_iterations = 1000
    l_lambda=1000
    threshold = 0.5
   # plt.show()
    print ("Starting gradient descent at thetha1 = {0}, thetha2= {1}, error = {2}".format(initial_thetha1, initial_thetha2,compute_error_for_graph_given_points(initial_thetha1,initial_thetha2, x1,x2,y)))
    print ("Running...")  
    [thetha1,thetha2] = gradient_descent_runner(x1,x2,y, initial_thetha1,initial_thetha2, learning_rate,l_lambda, num_iterations)    
    max_x=np.amax(x1_test)
    min_x=np.amin(x1_test)
    mean_x=np.mean(x1_test)
    for item in x1_test:
        feature_x= (item-mean_x)/(max_x-min_x)
        x_features.append(feature_x)
    
    #y=np.exp(b*feature_x)/(1+np.exp(b*feature_x))
    #print("value of feature x is:::",feature_x)
    #x_features.append(feature_x)
    #print("value of feature x after appending is:::",x_features)
    #x_array = np.array(x_features)
   # print("value of feature x after appending array is:::",x_array)
    plt.scatter(x_features, y_test)
    y_sigmoid=sigmoid(thetha1,thetha2,x1_test,x2_test)
    
    for item in y_sigmoid:
        if item > threshold :
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
    precision = true_pos/(true_pos+false_pos)
    recall = true_pos/(true_pos+false_neg)
    f1_score= 2*((precision*recall)/(precision+recall))
    print("precsion is::::",precision)
    print("Recall is::::",recall)
    print("f1-score is::::",f1_score)
    
    #print("x value is:::",x1)
    #print("y value is::::",y)
    plt.plot(x_features,y_predict, color='r')
    plt.show()
    print("After {0} iterations thetha1 = {1}, thetha2 = {2}, error = {3}".format(num_iterations, thetha1,thetha2, compute_error_for_graph_given_points(thetha1,thetha2, x1,x2,y)))

    
   
if __name__ == '__main__':
    run()

