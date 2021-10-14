import numpy as np

# Solve the least square problem by setting gradients w.r.t. weights to zeros
def MyLeastSquare(X,y):
    # placeholders, ensure the function runs
    w = np.array([1.0,-1.0])
    error_rate = 1.0

    # calculate the optimal weights based on the solution of Question 1
    w = np.matmul(np.matmul(np.linalg.matrix_power(np.matmul(X.transpose(), X), -1), X.transpose()), y)

    # compute the error rate
    badPredictions = 0
    t = 0
    while t < len(X):
        prediction = y[t] * np.matmul(w, X[t])
        if(prediction == 1):
            badPredictions +=1
        t+=1
    error_rate = badPredictions/len(X)

    return (w,error_rate)
