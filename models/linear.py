import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

######
#model=['linear regression', 'Rigde regression']
######



def linear(data):
#Casting 1D array in 2D
    x = np.array(data[['time','cases']])
    y = np.array(data['deaths'])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#check for more models

    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(x_train,y_train)

    coef = lin_reg.coef_
    y_pred = lin_reg.predict(x_test)
    print('The coefficients for [time cases] are : ',coef)

#Metrics

    mse = mean_squared_error(y_test, y_pred)
    print('The MSE is : ',mse)
    r2 = r2_score(y_test, y_pred)
    print('The R squared is : ',r2 )

#Plotting

    print('Saving plots to deaths_vs_time.png')
    plt.scatter(x_test[:,0], y_test,  color='black', label = 'Exact')
    plt.scatter(x_test[:,0], y_pred, color='blue', label = 'Linear model')
    plt.xlabel('Days passed since 31/12/2019')
    plt.ylabel('Deaths')
    plt.xticks()
    plt.yticks()
    plt.savefig('output_files/deaths_vs_time.png')

    print('Saving plots to deaths_vs_cases.png')
    plt.scatter(x_test[:,1], y_test,  color='black', label = 'Exact')
    plt.scatter(x_test[:,1], y_pred, color='blue', label = 'Linear model')
    plt.xlabel('Daily cases')
    plt.ylabel('Daily deaths')
    plt.xticks()
    plt.yticks()
    plt.savefig('output_files/deaths_vs_cases.png')
    return
