import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

######
#model=['linear regression', 'Rigde regression']
######



def linear(data):
    x = np.array(data[['time','cases']])
    y = np.array(data['deaths'])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#check for more methods
    metric = pd.DataFrame(data = np.zeros([1,2]) ,columns = ['squares', 'Ridge'])
    y_pred = pd.DataFrame(columns = ['squares', 'Ridge'])

#least squares

    least_squares = linear_model.LinearRegression()
    least_squares.fit(x_train,y_train)

    y_pred.squares = least_squares.predict(x_test).reshape(-1)
    metric.squares = r2_score(y_test, y_pred.squares)

#Ridge regression

    ridge = linear_model.RidgeCV(alphas = np.arange(1,11,1), cv = 5)
    ridge.fit(x_train, y_train)
    y_pred.Ridge = ridge.predict(x_test)
    metric.Ridge = r2_score(y_test, y_pred.Ridge)

#Output
    if metric.squares[0]>metric.Ridge[0]:
        model = 'squares'
        print('Linear regression with least squares : ')
        print('The coefficients for [time cases] are : ', least_squares.coef_)
        print('The accuracy of the model is : ', cross_validate(least_squares, x_train, y_train)['test_score'].mean())
        print('The MSE is : ',mean_squared_error(y_test, y_pred.squares))
        print('The R squared is : ',  metric.squares[0] )
    else :
        model = 'Ridge'
        print('Ridge linear regression : ')
        print('The coefficients for [time cases] are : ', ridge.coef_)
        print('The value of the regularization parameter is : ', ridge.alpha_)
        print('The MSE is : ',mean_squared_error(y_test, y_pred.Ridge))
        print('The R squared is : ',  metric.Ridge[0] )

#Plotting

    print('Saving plots to deaths.png')

    fig, (ax1, ax2) = plt.subplots(2, figsize = (10,10))
    ax1.scatter(x_test[:,0], y_test,  color='black', label = 'Exact')
    ax1.scatter(x_test[:,0], y_pred[model], color='blue', label = model)
    ax1.set(xlabel = 'Days passed since 31/12/2019', ylabel = 'Deaths')
    ax1.legend()

    ax2.scatter(x_test[:,1], y_test,  color='black', label = 'Exact')
    ax2.scatter(x_test[:,1], y_pred[model], color='blue', label = model)
    ax2.set(xlabel = 'Daily cases', ylabel = 'Daily deaths')
    ax2.legend()

    fig.savefig('output_files/deaths.png')

    return
