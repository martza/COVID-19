# Statistical Modelling of Covid-19 Data

## Introduction

The aim of this project is to model and predict the number of deaths in a region taking into account the correlation between daily new cases and daily new deaths over time.

## Package Description

The code takes as input a region and the target variable (cases, deaths). Computes the model parameters, the metrics and the prediction.  
* If the target is the cases, the code selects the model.
* If the target is the deaths, the code computes the parameters of a linear or non-linear two-dimensional model.
* If the user requests a prediction of the number of deaths at a given date, using the model for the cases over time, the code predicts the number of deaths (*not implemented yet*).

### Usage

The user needs to ``covid_19.py`` parsing different arguments:

1. ``-t`` for the target variable (``cases`` or ``deaths``). Default value is ``deaths``.
2. ``-m`` for model selection (``linear`` or ``non-linear``). This option applies only when deaths is the target. Default value is ``linear``.
3. ``-r`` for the region. Default value is ``all``, but one can parse the name of any country or continent, any countrycode or geoID.  

#####Examples:

* To request a model that describes the evolution of COVID-19 cases with time in Italy, you may use the following command line :

      $ python covid_19.py -t cases -r Italy  
  It returns :

      The model is :  Gaussian
      The R squared is : 0.5035300768867241

* To apply a linear model to the number of deaths in Italy, you may use the following command line :

      $ python covid_19.py -m linear -r Italy
  It returns:

      The model coefficients for [time cases] are :  [1.93497211 0.13237863]
      The MSE is :  18305.763810591132
      The R squared is :  0.754223894501798
      Saving plots to deaths_vs_time.png
      Saving plots to deaths_vs_cases.png


### Data

The dataset used in this package contains the number of daily new cases and new deaths due to COVID-19 by country and continent. It is updated on a daily basis.

### Statistical Models

* For the daily new cases the code fits three different models (linear, gaussian and logistic) and selects the model with the R squared score that is closest to 1.

* A linear model is used for mapping daily new deaths to daily new cases in time (*the implementation of a non-linear model is in progress*).

### Code Organization

```
COVID-19
├── covid_19.py
├── models
│   ├── prep_data.py
│   ├── model_cases.py
│   ├── linear.py
│   └── non-linear.py
├── output_files
│   └── linear.py
└── README.md
```

## Installation


## Acknowledgements

The [dataset](https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide) used in this package has been made publically available by the European Centre for Disease Prevention and Control (ECDC).

## Contributions
