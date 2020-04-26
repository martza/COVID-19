import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model

data = pd.read_csv('ECDC_COVID')
data.columns
#remove 2019 data
data = data[data['year'] == 2020]
#Select dataset1 for exploration at a 'global' level and based on 'daily' new cases and deaths
data1 = data[['dateRep','day','month','cases','deaths']]
#data1
global_data = data1.groupby(['dateRep']).agg(
            month = pd.NamedAgg(column = 'month', aggfunc = 'min'),
            day = pd.NamedAgg(column = 'day', aggfunc = 'min'),
            daily_global_cases = pd.NamedAgg(column = 'cases', aggfunc = 'sum'),
            daily_global_deaths = pd.NamedAgg(column = 'deaths', aggfunc = 'sum')
            )
global_data.columns
global_data.sort_values( by = ['month', 'day'])

global_data['time']=np.arange(116)
global_data

#explote correlation between global daily deaths and global daily cases with time
plt.scatter(global_data['daily_global_cases'],global_data['daily_global_deaths'])

sns.pairplot(global_data, hue= 'month', hue_order=None, palette= 'GnBu_d',
              vars= ['daily_global_cases', 'daily_global_deaths', 'time'],
             x_vars=None, y_vars=None, kind='scatter', diag_kind='auto',
             markers=None, height=2.5, aspect=1, corner=False, dropna=True,
             plot_kws=None, diag_kws=None, grid_kws=None, size=None)

y = np.array(global_data['daily_global_deaths'])
x = np.array(global_data[['daily_global_cases','time']])
lin_reg = linear_model.LinearRegression()
lin_reg.fit(x,y)
lin_reg.coef_
