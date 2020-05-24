import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.covariance import EllipticEnvelope

######
#model=['linear regression', 'Rigde regression']
######



def linear(data):
#Casting 1D array in 2D
    x = np.array(data[['time','cases']])
    y = np.array(data['deaths'])

#Removing outliers
    cov = EllipticEnvelope(random_state=0).fit(y.reshape(-1,1))
    cov.predict(y.reshape(-1,1))
    x = x[cov.predict(y.reshape(-1,1)) == 1]
    y = y[cov.predict(y.reshape(-1,1)) == 1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#check for more models

    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(x_train,y_train)

    accuracy = cross_validate(lin_reg, x_train, y_train)['test_score'].mean()

    y_pred = lin_reg.predict(x_test)

#Output
    print('The coefficients for [time cases] are : ', lin_reg.coef_)
    print('The accuracy of the model is : ', accuracy)
    mse = mean_squared_error(y_test, y_pred)
    print('The MSE is : ',mse)
    r2 = r2_score(y_test, y_pred)
    print('The R squared is : ',r2 )

#Plotting

    print('Saving plots to deaths.png')

    fig, (ax1, ax2) = plt.subplots(2, figsize = (10,10))

    ax1.scatter(x_test[:,0], y_test,  color='black', label = 'Exact')
    ax1.scatter(x_test[:,0], y_pred, color='blue', label = 'Linear model')
    ax1.set(xlabel = 'Days passed since 31/12/2019', ylabel = 'Deaths')
    ax1.legend()

    ax2.scatter(x_test[:,1], y_test,  color='black', label = 'Exact')
    ax2.scatter(x_test[:,1], y_pred, color='blue', label = 'Linear model')
    ax2.set(xlabel = 'Daily cases', ylabel = 'Daily deaths')
    ax2.legend()

    fig.savefig('output_files/deaths.png')

    return
