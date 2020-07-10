import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn import linear_model
from sklearn.pipeline import Pipeline


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

global_data = global_data[['time','cases','deaths']]

scaler = StandardScaler()
scaler.fit(global_data[['time']])
transformed_data = scaler.transform(global_data[['time']])
transformed_data.reshape(-1)
scaler.inverse_transform(transformed_data)
scaled_data[5]
scaler_x = StandardScaler()
scaler_y = StandardScaler()
y = np.array(global_data['deaths'])
x = np.array(global_data[['time','cases']])



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

y_pred = pd.DataFrame()
metric = pd.DataFrame()
y_pred_ridge = pd.DataFrame()
metric_ridge = pd.DataFrame()
alpha = pd.DataFrame()

for i in np.arange(1,10) :
    pipe = Pipeline([('poly',PolynomialFeatures(degree=i)),
                     ('least_squares',linear_model.LinearRegression())])
    pipe.fit(x_train, y_train)
    y_pred[i] = pipe.predict(x_test)
    metric[i] = r2_score(y_test, np.array(y_pred[i])).reshape(-1)

for i in np.arange(1,10) :
    pipe = Pipeline([('poly',PolynomialFeatures(degree=i)),
                     ('ridge',linear_model.RidgeCV(alphas = np.arange(1,11,1), cv = 5))])
    pipe.fit(x_train, y_train)
    y_pred_ridge[i] = pipe.predict(x_test)
    metric_ridge[i] = r2_score(y_test, np.array(y_pred[i])).reshape(-1)
    alpha[i] = ridge.alpha_.reshape(-1)
alpha
y_pred_ridge
metric_ridge
metric
degree = metric.idxmax(axis = 1)[0]
degree_ridge = metric_ridge.idxmax(axis = 1)[0]

if (metric[degree][0]<metric_ridge[degree_ridge][0]):
    model = 'Ridge'
    y_pred = y_pred_rigde
    metric = metric_ridge
    degree = degree_ridge
else :
    model = 'Least squares'
model


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
