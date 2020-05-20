import pandas as pd
import numpy as np
import sys

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
#Output is the dataset
###############################################################################
    reg = region.casefold()
#todo: make the comparison case insensitive
    if (reg in country_names):
        data = full_data[full_data['countriesAndTerritories']==reg][['dateRep', 'cases', 'deaths']]
    elif (reg in country_codes):
        data = full_data[full_data['countryterritoryCode']==reg][['dateRep', 'cases', 'deaths']]
    elif (reg in continent_names):
        data = full_data[full_data['continentExp']==reg].groupby(['dateRep'], as_index = False).sum()
    elif (reg == 'all'):
        data = full_data.groupby(['dateRep'], as_index = False).sum()
    else:
        print('Invalid country name')
        sys.exit(-1)

    data = data.sort_values(by = 'dateRep')
    data['time']=data.index
    return data
