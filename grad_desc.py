#Gradient Descent for Linear Regression

#Y = WX + b

#loss = (y-yhat)**2/N

import numpy as np
#Initializze some parameters
#we don't know what W and b are
#Based on the x and y data points, we can calculate W and b

x = np.random.randn(10,1)
y = 2*x + np.random.rand() #y=2x+b
#Parameters
w = 0.0
b = 0.0
#Hyperparameteres
learning_rate = 0.01  #How fast the algorithm learns
#Iteratively, we will find out what are the values of w and b, Ideally when we run through Gradient Descent, we should be able to get w=2 and the particular b value{np.random.rand()}

#create Gradient Descent fuction
def grad_descent(x, y, w, b, learning_rate):
    #initialize the partial derivative - calculating l with respect to each of the parameters
    dldw = 0.0
    dldb = 0.0
    N = x.shape[0]
    #loss = (y-(wx+b))**2
    #looping through the samples inside our data,
    for xi,yi in zip(x,y):
        dldw += -2*xi*yi*(yi-((w*xi)+b))
        dldb += -2*(yi-((w*xi)+b))
    #update w and b parameters
    w = w - learning_rate*(1/N)*dldw
    b = b - learning_rate*(1/N)*dldb
    return w,b



#Iteratively make updates to the gradient

for epoch in range(400):
    w, b = grad_descent(x, y, w, b, learning_rate)
    yhat = (w*x) + b
    loss = np.divide(np.sum((y-yhat)**2,axis=0),x.shape[0])
    print(f'{epoch} loss is {loss}, parameters w:{w}, b:{b}')
    #Run Gradient Descent
    