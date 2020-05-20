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
    x = np.array(data['cases']).reshape(-1,1)
    y = np.array(data['deaths']).reshape(-1,1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#check for more models

    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(x_train,y_train)

    coef = lin_reg.coef_
    y_pred = lin_reg.predict(x_test)
    print('The coefficients are : ',coef)

#calculate metrics

    mse = mean_squared_error(y_test, y_pred)
    print('The mse is : ',mse)
    r2 = r2_score(y_test, y_pred)
    print('The r2 score is : ',r2 )

#make the plot

    print('Saving plot to linear_model.png')
    plt.scatter(x_test, y_test,  color='black')
    plt.plot(x_test, y_pred, color='blue', linewidth=3)
    plt.title('linear model')
    plt.xlabel('global daily cases')
    plt.ylabel('global daily deaths')
    plt.xticks()
    plt.yticks()
    plt.savefig('output_files/linear_model.png')
    return
