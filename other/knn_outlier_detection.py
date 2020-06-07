import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, scale
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors


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

set = np.array(global_data[['time','cases','deaths']])
###############################################################################
#Check for outliers
###############################################################################
scaled_set = scale(set)

max_outliers_portion = 0.1

max_outliers = int(max_outliers_portion * len(set))

knn = NearestNeighbors(radius =0.5 , metric='minkowski', p = 3)
knn.fit(scaled_set)

neigh_dist, neigh_ind = knn.radius_neighbors(scaled_set, return_distance=True)
rho = []
for i in np.arange(0,len(scaled_set)):
    rho.append(len(neigh_ind[i]))
nns = 1
while (sum(np.array(rho)<=nns)<= max_outliers):
    nns = nns+1
nns

plt.scatter(scaled_set[:,0],scaled_set[:,1], label = 'outliers', color = 'red')
plt.scatter(scaled_set[np.array(rho)>3,0],scaled_set[np.array(rho)>3,1], label = 'clean')
plt.legend()
plt.show()

plt.scatter(scaled_set[:,0],scaled_set[:,2], label = 'outliers', color = 'red')
plt.scatter(scaled_set[np.array(rho)>3,0],scaled_set[np.array(rho)>3,2], label = 'clean')
plt.legend()
plt.show()

plt.scatter(scaled_set[:,1],scaled_set[:,2], label = 'outliers', color = 'red')
plt.scatter(scaled_set[np.array(rho)>3,1],scaled_set[np.array(rho)>3,2], label = 'clean')
plt.legend()
plt.show()

np.std(set, axis = 0)
np.std(set[np.array(rho)>2], axis = 0)
