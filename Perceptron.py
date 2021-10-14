import numpy as np

# Implement the Perceptron algorithm
def MyPerceptron(X,y,w0=[1.0,-1.0]):
    k = 0 # initialize variable to store number of iterations it will take
          # for your perceptron to converge to a final weight vector
    w=w0
    w1=[0.0,0.0]
    error_rate = 1.00

    # loop until convergence (w does not change at all over one pass)
    # or until max iterations are reached
    # (current pass w ! = previous pass w), then do:
    #
    while((w != w1).all()):
        w1=w
        # for each training sample (x,y):
        t = 0
        while t < len(X):
            # if actual target y does not match the predicted target value, update the weights
            # calculate the number of iterations as the number of updates
            if((y[t] * np.matmul(w, X[t])) < 0):
                w = w + np.dot(y[t], X[t])
                k+=1
            t+=1
    print("Iterations: " + str(k))


        

    # make prediction on the csv dataset using the feature set
    # Note that you need to convert the raw predictions into binary predictions using threshold 0
    badPredictions = 0
    t = 0
    while t < len(X):
        prediction = y[t] * np.matmul(w, X[t])
        if(prediction == 1):
            badPredictions +=1
        t+=1


    # compute the error rate
    # error rate = ( number of prediction ! = y ) / total number of training examples
    error_rate = badPredictions/len(X)

    return (w, k, error_rate)
