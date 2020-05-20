import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PolynomialFeatures

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

data = pd.read_csv('https://opendata.ecdc.europa.eu/covid19/casedistribution/csv')
data['dateRep']=pd.to_datetime(data['dateRep'], infer_datetime_format = True)
#Select dataset 1 for exploration at a 'global' level and based on 'daily' new cases and deaths
data1 = data[['dateRep', 'cases', 'deaths']]
global_data = data1.groupby(['dateRep'], as_index = False).sum()
global_data.sort_values( by = ['dateRep'],inplace=True)
global_data['time']=global_data.index
global_data = global_data[global_data['cases']>0]

################################################################################
#looking at the data it seems to follow a gaussian or logistic process
#two types of models that one can try:
#1 exponential : log(y) = a x + b ---> y = exp(a x + b)
#2 gaussian : log(y) = a x**2 + b x + c
#3 logistic : y= 1/(1+e**(-x))
################################################################################

y = np.array(global_data['cases']).reshape(-1,1)
x = np.array(global_data['time']).reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

######################################
# model 1
######################################
tt = TransformedTargetRegressor(regressor=LinearRegression(),func=np.log, inverse_func=np.exp)
tt.fit(x_train,y_train)
r2_score = tt.score(x_train,y_train)
r2_score
y_pred = tt.predict(x_test)

plt.scatter(x_test,y_pred)
plt.scatter(x_train,y_train)
plt.show()
######################################
# model 2
######################################

tt1 = TransformedTargetRegressor(regressor=LinearRegression(),func=np.log, inverse_func=np.exp)

poly = PolynomialFeatures(degree=2)

x_train_nl = poly.fit_transform(x_train)
x_test_nl = poly.fit_transform(x_test)

tt1.fit(x_train_nl,y_train)
y_pred1 = tt1.predict(x_test_nl)


tt1.score(x_train_nl,y_train)

plt.scatter(x_test,y_pred1)
plt.scatter(x_train,y_train)
plt.show()

######################################
# model 3
######################################

from scipy import optimize

def sigmoid(x, a1, c1, d1):
    g = c1*np.exp((x-a1)/d1)/(np.exp((x-a1)/d1)+1.0)
    return g
def mse(y,y1):
    m = np.sum((y-y1)**2/len(y))
    return m
def r2(y,y_new):
    r = 1-np.sum((y-y_new)**2)/sum((y_new-mse(y,y_new))**2)
    return r

y1 = np.array(global_data['cases'])
x1 = np.array(global_data['time'])

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.2, random_state=42)

popt, pcov = optimize.curve_fit(sigmoid, x1_train, y1_train)
y1_pred = sigmoid(x1_test,*popt)

mse(y1_pred,y1_test)
r2(y1_test,y1_pred)

plt.scatter(x1_test,y1_test, label = 'exact')
plt.scatter(x1_test,y1_pred, label = 'transform')
plt.legend()
plt.show()


#for the plot generate more data points with the model
x_time_new = np.arange(1,150).reshape(-1,1)
y_cases_new = sigmoid(x_time_new,*popt1)
y_deaths_pred_new = lin_reg.predict(np.column_stack((y_cases_new,x_time_new)))

plt.scatter('time', 'deaths', data=global_data,  color='black')
plt.scatter(x_time_new[:,0], y_deaths_pred_new, color='blue')
plt.title('linear model')
plt.xlabel('time')
plt.ylabel('global daily cases')
plt.xticks()
plt.yticks()
#plt.savefig('linear_model.png')
plt.show()
