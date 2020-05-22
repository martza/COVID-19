import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PolynomialFeatures
from scipy import optimize
###############################################################################
#Some functions
###############################################################################
def sigmoid(x, a1, c1, d1):
    g = c1*np.exp((x-a1)/d1)/(np.exp((x-a1)/d1)+1.0)
    return g
##############################################################################
#This is a function fitting the number of cases with a linear model. This
#function can be used for prediction.
#y = number of cases
#x = time
#Returns:
#model errors (metrics)
#data
#scatter plot with data and the model
###############################################################################


###############################################################################
#Load the dataset and clean it
###############################################################################

def model_cases(data):
################################################################################
#looking at the data it seems to follow a gaussian or logistic curve
#two types of models that one can try:
#1 exponential : log(y) = a x + b ---> y = exp(a x + b)
#2 gaussian : log(y) = a x**2 + b x + c
#3 logistic : y= 1/(1+e**(-x))
################################################################################
    y = np.array(data['cases']).reshape(-1,1)
    x = np.array(data['time']).reshape(-1,1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#Arrays for storing metrics and predictions
    metric = pd.DataFrame(data = np.zeros([1,3]) ,columns = ['Linear', 'Gaussian', 'Logistic'])
    y_pred = pd.DataFrame(columns = ['Linear', 'Gaussian', 'Logistic'])

######################################
# model 1 Linear
######################################

    tt_lin = TransformedTargetRegressor(regressor=LinearRegression(),func=np.log, inverse_func=np.exp)
    tt_lin.fit(x_train,y_train)
    metric['Linear'] = tt_lin.score(x_train,y_train)
    y_pred['Linear'] = tt_lin.predict(x_test).reshape(-1)

######################################
# model 2 Gaussian
######################################

    tt_gauss = TransformedTargetRegressor(regressor=LinearRegression(),func=np.log, inverse_func=np.exp)
    poly = PolynomialFeatures(degree=2)
    x_train_gauss = poly.fit_transform(x_train)
    x_test_gauss = poly.fit_transform(x_test)
    tt_gauss.fit(x_train_gauss,y_train)
    y_pred['Gaussian'] = tt_gauss.predict(x_test_gauss)
    metric['Gaussian'] = tt_gauss.score(x_train_gauss,y_train)

######################################
# model 3 Logistic
######################################

    popt, pcov = optimize.curve_fit(sigmoid, x_train.reshape(-1), y_train.reshape(-1))
    y_pred['Logistic'] = sigmoid(x_test,*popt)
    metric['Logistic'] = r2_score(y_test, y_pred['Logistic'])

################################################################################
#Comparison among models and plots
################################################################################

    r2_value = metric.max(axis = 1)[0]
    model = metric.idxmax(axis = 1)[0]
    print('The model is : ',model)
    print('The R squared is :', r2_value)
    plt.scatter(x_test,y_test, label = 'Exact')
    plt.scatter(x_test,y_pred[model], label = model)
    plt.ylabel('Cases')
    plt.xlabel('Days passed since 31/12/2019')
    plt.legend()
    plt.show()

    return
