import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def linear(data):

    '''
    linear(x): A function that takes the dataset and trains a linear model
    Returns model parameters, metrics and graphs

        Model: y = a x1 + b x2 + c
        Target variable: y = daily new deaths
        Dimensions: (x1,x2) = (time since 31/12/2019, daily new cases)
        Regression methods: [least squares, Ridge regression]
        Parameters: (a,b,c)

        Steps:
        * Train-test splitting
        * Fitting
        * Target prediction on the test set
        * Select the method with the highest R squared and print information
        * Graphical comparison between test values and prediction
    '''

    # Dataframes for keeping the metrics and predictions of the two methods
    metric = pd.DataFrame(data = np.zeros([1,2]) ,columns = ['squares', 'Ridge'])
    y_pred = pd.DataFrame(columns = ['squares', 'Ridge'])

    x = np.array(data[['time','cases']])                                        # Dimensions
    y = np.array(data['deaths'])                                                # Target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                        random_state=42)        # Train and test arrays
    # Fitting models
    least_squares = linear_model.LinearRegression()                             # Linear regression with least squares
    least_squares.fit(x_train,y_train)                                          # Fitting
    y_pred.squares = least_squares.predict(x_test).reshape(-1)                  # Prediction
    metric.squares = r2_score(y_test, y_pred.squares)                           # R2 score

    # Linear regression with Ridge coefficients and cross-validation
    ridge = linear_model.RidgeCV(alphas = np.arange(1,11,1), cv = 5)            # Ridge model
    ridge.fit(x_train, y_train)                                                 # Fitting
    y_pred.Ridge = ridge.predict(x_test)                                        # Prediction
    metric.Ridge = r2_score(y_test, y_pred.Ridge)                               # R2 score

    # Model selection based on the the R squared score
    if metric.squares[0]>metric.Ridge[0]:
        model = 'squares'
        print('Linear regression with least squares : ')
        print('The coefficients for [time cases] are : ', least_squares.coef_)
        print('The MSE is : ',mean_squared_error(y_test, y_pred.squares))
        print('The R squared is : ',  metric.squares[0] )
    else :
        model = 'Ridge'
        print('Ridge linear regression : ')
        print('The coefficients for [time cases] are : ', ridge.coef_)
        print('The value of the regularization parameter is : ', ridge.alpha_)
        print('The MSE is : ',mean_squared_error(y_test, y_pred.Ridge))
        print('The R squared is : ',  metric.Ridge[0] )

    # Graphical comparison between test values and prediction
    print('Saving plots to deaths.png')

    fig, (ax1, ax2) = plt.subplots(2, figsize = (10,10))

    # Plot w.r.t. the time variable
    ax1.scatter(x_test[:,0], y_test,  color='black', label = 'Exact')
    ax1.scatter(x_test[:,0], y_pred[model], color='blue', label = model)
    ax1.set(xlabel = 'Days passed since 31/12/2019', ylabel = 'Deaths')
    ax1.legend()

    # Plot w.r.t. the daily new cases variable
    ax2.scatter(x_test[:,1], y_test,  color='black', label = 'Exact')
    ax2.scatter(x_test[:,1], y_pred[model], color='blue', label = model)
    ax2.set(xlabel = 'Daily cases', ylabel = 'Daily deaths')
    ax2.legend()

    fig.savefig('output_files/deaths.png')

    return


def non_linear(data) :

    '''
    non_linear(x): A function that takes the dataset and trains a non-linear model
    Returns model parameters, metrics and graphs

        Model: y = P(x1,x2)
        Target variable: y = daily new deaths
        Dimensions: (x1,x2) = (time since 31/12/2019, daily new cases)
        Regression methods: [least squares, Ridge regression]

        Steps:
        * Train-test splitting
        * Standardisation of the variables
        * Fitting polynomials
        * Target prediction on the test set
        * Choosing the polynomial with the largest R squared for each method
        * Select the method with the highest R squared and print information
        * Graphical comparison between test values and prediction
    '''

    # Initialisation of dataframes for storing the results from different polynomials
    y_pred = pd.DataFrame()                                                     # predictions for lsq
    metric = pd.DataFrame()                                                     # metric for lsq
    y_pred_ridge = pd.DataFrame()                                               # predictions for Ridge
    metric_ridge = pd.DataFrame()                                               # metric for Ridge
    alpha = pd.DataFrame()                                                      # regularization values for Ridge

    x = np.array(data[['time','cases']])                                        # Dimensions
    y = np.array(data['deaths'])                                                # Target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Evaluate predictions from polynomials of degrees (1:10)
    for i in np.arange(1,10) :
        pipe_lsq = Pipeline([('scaler',StandardScaler()),
                             ('poly',PolynomialFeatures(degree=i)),
                             ('least_squares',linear_model.LinearRegression())])
        pipe_lsq.fit(x_train, y_train)                                          # transforms only x
        y_pred[i] = pipe_lsq.predict(x_test)
        metric[i] = r2_score(y_test, np.array(y_pred[i])).reshape(-1)

        pipe_ridge = Pipeline([('scaler',StandardScaler()),
                               ('poly',PolynomialFeatures(degree=i)),
                               ('ridge',linear_model.RidgeCV(alphas = np.arange(1,11,1), cv = 5))])
        pipe_ridge.fit(x_train, y_train)
        y_pred_ridge[i] = pipe_ridge.predict(x_test)
        metric_ridge[i] = r2_score(y_test, np.array(y_pred[i])).reshape(-1)
        alpha[i] = pipe_ridge.named_steps['ridge'].alpha_.reshape(-1)

    # Choose the polynomial of the degree that gives the largest R squared for each method
    degree = metric.idxmax(axis = 1)[0]
    degree_ridge = metric_ridge.idxmax(axis = 1)[0]

    # Choose the method that gives the largest R squared
    if (metric[degree][0]<metric_ridge[degree_ridge][0]):
        model = 'Ridge'
        y_pred_out = scaler_y.inverse_transform(y_pred_ridge[degree_ridge])
        metric_out = r2_score(scaler_y.inverse_transform())
        degree = degree_ridge
        print('Ridge polynomial regression : ')
        print('The degree of the polynomial is : ', degree)
        print('The value of the regularization parameter is : ', alpha[degree][0])
        print('The MSE is : ',mean_squared_error(y_test, y_pred[degree]))
        print('The R squared is : ',  metric[degree][0])
    else :
        model = 'Least squares'
        print('Polynomial regression with least squares : ')
        print('The degree of the polynomial is : ', degree)
        print('The MSE is : ',mean_squared_error(y_test, y_pred[degree]))
        print('The R squared is : ',  metric[degree][0] )

    # Graphical comparison between test values and prediction
    print('Saving plots to deaths.png')

    fig, (ax1, ax2) = plt.subplots(2, figsize = (10,10))

    # Plot w.r.t. the time variable
    ax1.scatter(x_test[:,0], y_test,  color='black', label = 'Exact')
    ax1.scatter(x_test[:,0], y_pred[degree], color='blue', label = model)
    ax1.set(xlabel = 'Days passed since 31/12/2019', ylabel = 'Deaths')
    ax1.legend()

    # Plot w.r.t. the daily new cases variable
    ax2.scatter(x_test[:,1], y_test,  color='black', label = 'Exact')
    ax2.scatter(x_test[:,1], y_pred[degree], color='blue', label = model)
    ax2.set(xlabel = 'Daily cases', ylabel = 'Daily deaths')
    ax2.legend()

    fig.savefig('output_files/deaths.png')

    return
