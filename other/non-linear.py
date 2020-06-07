import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

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
from sklearn.pipeline import Pipeline

clf = LocalOutlierFactor(n_neighbors=2, contamination = 0.1)
x = x[clf.fit_predict(y.reshape(-1,1)) == 1]
y = y[clf.fit_predict(y.reshape(-1,1)) == 1]

plt.scatter(x[:,0],y, label = 'outliers')
plt.legend()
plt.show()
plt.scatter(x[:,1],y, label = 'outliers')
plt.legend()
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

y_pred = pd.DataFrame()
for i in np.arange(1,2) :
    pipe = Pipeline([('poly',PolynomialFeatures(degree=i)),('least_squares',linear_model.LinearRegression())])
    pipe.fit(x_train, y_train)
    y_pred[i] = pipe.predict(x_test)
y_pred

x_train_nl = poly.fit_transform(x_train)
x_test_nl = poly.fit_transform(x_test)

ridge = linear_model.RidgeCV(alphas = np.arange(1,11,1), cv = 5)
ridge.fit(x_train_nl, y_train)
ridge.coef_
ridge.alpha_
y_pred1 = ridge.predict(x_test_nl)
ridge.score(x_test_nl, y_test)
r2_score(y_test, y_pred1)

lin_reg = linear_model.LinearRegression()
lin_reg.fit(x_train_nl,y_train)
accuracy = cross_validate(lin_reg, x_train_nl, y_train)['test_score'].mean()
coef = lin_reg.coef_
coef
y_pred = lin_reg.predict(x_test_nl)
mse = mean_squared_error(y_test, y_pred)
r2_score(y_test, y_pred)
y_pred
#for the plot generate more data points with the model
plt.scatter(x[:,1], y, color='black')
plt.scatter(x_test[:,1], y_pred, color='blue')
plt.scatter(x_test[:,1], y_pred1, color='green')
plt.title('linear model')
plt.xlabel('global daily cases')
plt.ylabel('global daily deaths')
plt.xticks()
plt.yticks()
plt.show()

plt.scatter(x[:,0], y, color='black')
plt.scatter(x_test[:,0], y_pred, color='blue')
plt.scatter(x_test[:,0], y_pred1, color='green')
plt.title('linear model')
plt.xlabel('time (days)')
plt.ylabel('global daily deaths')
plt.xticks()
plt.yticks()
plt.show()
