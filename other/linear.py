import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, r2_score
###############################################################################
#This is a function fitting the dataset with a linear model
#y=f({x_i},n)=Sum_i^n(a_i*x_i)
#y = number of deaths
#x = [cases, time]
#Returns:
#Parameters {a_i} if requested
#model errors (metrics)
#scatter plot with data and the model
#USER Input:
#Region = ['Country', 'Continent', 'All']
#
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

y = np.array(global_data['deaths'])
x = np.array(global_data[['time','cases']])
###############################################################################
#Check for outliers
###############################################################################
plt.scatter(x[:,0],y)
plt.scatter(x[:,1],y)

from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

clf = LocalOutlierFactor(n_neighbors=2)
x1 = x[clf.fit_predict(y.reshape(-1,1)) == 1]
y1 = y[clf.fit_predict(y.reshape(-1,1)) == 1]

clf1 = OneClassSVM().fit(y.reshape(-1,1))
clf1.predict(y.reshape(-1,1))
x2 = x[clf1.predict(y.reshape(-1,1)) == 1]
y2 = y[clf1.predict(y.reshape(-1,1)) == 1]

cov = EllipticEnvelope(random_state=0).fit(y.reshape(-1,1))
cov.predict(y.reshape(-1,1))
x = x[cov.predict(y.reshape(-1,1)) == 1]
y = y[cov.predict(y.reshape(-1,1)) == 1]

plt.scatter(x[:,0],y, label = 'outliers')
plt.scatter(x1[:,0],y1)
#plt.scatter(x2[:,0],y2)
plt.scatter(x3[:,0],y3)
plt.legend()
plt.show()
plt.scatter(x[:,1],y, label = 'outliers')
plt.scatter(x1[:,1],y1)
#plt.scatter(x2[:,1],y2)
plt.scatter(x3[:,1],y3)
plt.legend()
plt.show()
#Local  Outlier factor and Elliptic Envelope behave the best

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#Use Ridge regression

ridge = linear_model.RidgeCV(alphas = np.arange(1,11,1), cv = 5)
ridge.fit(x_train, y_train)
ridge.coef_
ridge.alpha_
y_pred1 = ridge.predict(x_test)
ridge.score(x_test, y_test)
r2_score(y_test, y_pred1)

lin_reg = linear_model.LinearRegression()
lin_reg.fit(x_train,y_train)
accuracy = cross_validate(lin_reg, x_train, y_train)['test_score'].mean()
coef = lin_reg.coef_
coef
y_pred = lin_reg.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2_score(y_test, y_pred)

print('The coefficients are : ',coef)
print('The mse is : ',mse)
print('The r2 score is : ',r2 )
global_data['deaths']
#for the plot generate more data points with the model
y_plot = lin_reg.predict(x)
plt.scatter(x[:,1], y, color='black')
plt.scatter(x_test[:,1], y_pred, color='blue')
plt.scatter(x_test[:,1], y_pred1, color='green')
plt.title('linear model')
plt.xlabel('global daily cases')
plt.ylabel('global daily deaths')
plt.xticks()
plt.yticks()
#plt.savefig('linear_model.png')
plt.show()

plt.scatter(x[:,0], y, color='black')
plt.scatter(x_test[:,0], y_pred, color='blue')
plt.scatter(x_test[:,0], y_pred1, color='green')
plt.title('linear model')
plt.xlabel('time (days)')
plt.ylabel('global daily deaths')
plt.xticks()
plt.yticks()
#plt.savefig('linear_model.png')
plt.show()
###############################################################################
#make a dataset cases(time).
#fit with a gaussian model of the form logf(x)=ax^2+bx+c
from scipy import optimize

def gauss(x, a, b, c, d):
    f = c*np.exp(-(x-a)**2/b)+d
    return f
def sigmoid(x, a1, c1, d1):
    g = c1*np.exp((x-a1)/d1)/(np.exp((x-a1)/d1)+1.0)
    return g

y1 = np.array(global_data['cases'])
area = sum(y1)
area
y1_normalized = y1/area

x1 = np.array(global_data['time'])

popt1, pcov1 = optimize.curve_fit(sigmoid, x1, y1)
pcov1
plt.scatter(x1,y1, label = 'exact')
plt.plot(x1,sigmoid(x1,*popt1), label = 'transform1')
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
