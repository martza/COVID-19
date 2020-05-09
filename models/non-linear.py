import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

data = pd.read_csv('https://opendata.ecdc.europa.eu/covid19/casedistribution/csv')
data.columns
#remove 2019 data
data = data[data['year'] == 2020]
#Select dataset 1 for exploration at a 'global' level and based on 'daily' new cases and deaths
data1 = data[['dateRep','day','month','cases','deaths']]

global_data = data1.groupby(['dateRep']).agg(
            month = pd.NamedAgg(column = 'month', aggfunc = 'min'),
            day = pd.NamedAgg(column = 'day', aggfunc = 'min'),
            daily_global_cases = pd.NamedAgg(column = 'cases', aggfunc = 'sum'),
            daily_global_deaths = pd.NamedAgg(column = 'deaths', aggfunc = 'sum')
            )
global_data.columns
global_data.sort_values( by = ['month', 'day'])
nrows = len(global_data.index)
global_data['time']=np.arange(nrows)
#global_data
#explote correlation between global daily deaths and global daily cases with time
plt.scatter(global_data['daily_global_cases'],global_data['daily_global_deaths'])
y = np.array(global_data['daily_global_deaths'])
x = np.array(global_data[['daily_global_cases']])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
poly = PolynomialFeatures(degree=2)
x_train_nl = poly.fit_transform(x_train)
x_test_nl = poly.fit_transform(x_test)
lin_reg = linear_model.LinearRegression()
lin_reg.fit(x_train_nl,y_train)
lin_reg.coef_
y_pred = lin_reg.predict(x_test_nl)
mean_squared_error(y_test, y_pred)
r2_score(y_test, y_pred)

plt.scatter(x_test, y_test,  color='black')
plt.scatter(x_test, y_pred, color='blue')
plt.title('non-linear model')
plt.xlabel('global daily cases')
plt.ylabel('global daily deaths')
plt.xticks()
plt.yticks()
plt.savefig('non-linear_model.png')
plt.show()
