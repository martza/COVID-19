import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import scale
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt

full_data = pd.read_csv('https://opendata.ecdc.europa.eu/covid19/casedistribution/csv', encoding = 'utf_8')

#columns = ['dateRep', 'day', 'month', 'year', 'cases', 'deaths','countriesAndTerritories',
#'geoId', 'countryterritoryCode','popData2018', 'continentExp']

#cleaning the full dataset
full_data['dateRep']=pd.to_datetime(full_data['dateRep'], infer_datetime_format = True)
numeric = ['int64','float64']
length = len(full_data.columns)

for i in range(0, length):
    if (full_data.dtypes[i] in numeric):
        full_data[full_data.columns[i]].fillna(0, inplace = True)
    else:
        full_data[full_data.columns[i]].dropna(axis = 'index', inplace = True)

#make strings case insensitive
region_names = ['countriesAndTerritories','countryterritoryCode','geoId','continentExp']
for name in region_names:
    full_data[name] = full_data[name].str.casefold()

#get the list of countries and country codes
country_names = full_data['countriesAndTerritories'].unique()
country_codes = full_data['countryterritoryCode'].unique()
geoids = full_data['geoId'].unique()
continent_names = full_data['continentExp'].unique()

def dataset(region):
###############################################################################
#This function is preparing the dataset.
#Input takes is a country/country_code/continent/all
#Output is the dataset ['dateRep', 'time', 'cases', 'deaths']
###############################################################################
    reg = region.casefold()
#todo: make the comparison case insensitive
    if (reg in country_names):
        data = full_data[full_data['countriesAndTerritories']==reg][['dateRep', 'cases', 'deaths']]
    elif (reg in country_codes):
        data = full_data[full_data['countryterritoryCode']==reg][['dateRep', 'cases', 'deaths']]
    elif (reg in continent_names):
        data = full_data[full_data['continentExp']==reg][['dateRep', 'cases', 'deaths']].groupby(['dateRep'], as_index = False).sum()
    elif (reg == 'all'):
        data = full_data[['dateRep', 'cases', 'deaths']].groupby(['dateRep'], as_index = False).sum()
    else:
        print('Invalid country name')
        sys.exit(-1)

    data.sort_values(by = 'dateRep',inplace = True)
    data.reset_index(drop = True, inplace = True)
    data['time']=data.index
    data = data[data['cases']>0] #remove zeros for transforming with log
    data = data[['time','cases','deaths']]
    return data

def outlier_detection(data, outliers_portion, method) :
#Standardization using the standard deviation
    scaled_data = scale(np.array(data))
    if (method == 'knn') :
# This is a method to detect outliers by counting the number of neighbors within a distance from a point given by a radius
# rho is the number of neighboors of a data point within a given radius
# nns is the maximum number of neighbors of outliers
        knn = NearestNeighbors(radius =0.5 , metric='minkowski', p = 3)
        knn.fit(scaled_data)
        neigh_dist, neigh_ind = knn.radius_neighbors(scaled_data, return_distance=True)
        rho = []
        for i in np.arange(0,len(scaled_data)):
            rho.append(len(neigh_ind[i]))
        nns = 0
        max_outliers = int(outliers_portion * len(data))
        while (sum(np.array(rho) <= nns+1) <= max_outliers):
            nns = nns+1
        clean_data = scaled_data[np.array(rho) > nns]
        out_data = data[np.array(rho) > nns]
    elif (method == 'LocalOutlierFactor') :
        clf = LocalOutlierFactor(n_neighbors=30, p=3, contamination = outliers_portion)
        clean_data = scaled_data[clf.fit_predict(scaled_data) == 1]
        out_data = data[clf.fit_predict(scaled_data) == 1]
    elif (method == 'EllipticEnvelope') :
        cov = EllipticEnvelope(random_state=0, contamination = outliers_portion).fit(scaled_data)
        clean_data = scaled_data[cov.predict(scaled_data) == 1]
        out_data = data[cov.predict(scaled_data) == 1]
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize = (10,10))

    ax1.scatter(scaled_data[:,0],scaled_data[:,1], label = 'outliers', color = 'red')
    ax1.scatter(clean_data[:,0],clean_data[:,1], label = 'clean')
    ax1.set(xlabel = 'Days passed since 31/12/2019', ylabel = 'Cases')
    ax1.legend()

    ax2.scatter(scaled_data[:,0],scaled_data[:,2], label = 'outliers', color = 'red')
    ax2.scatter(clean_data[:,0],clean_data[:,2], label = 'clean')
    ax2.set(xlabel = 'Days passed since 31/12/2019', ylabel = 'Deaths')
    ax2.legend()

    ax3.scatter(scaled_data[:,1],scaled_data[:,2], label = 'outliers', color = 'red')
    ax3.scatter(clean_data[:,1],clean_data[:,2], label = 'clean')
    ax3.set(xlabel = 'Daily cases', ylabel = 'Daily deaths')
    ax3.legend()

    fig.savefig('output_files/outliers.png')

    return out_data
