# Statistical Modelling of Covid-19 Data

## Introduction

The aim of this project is to model and predict the number of deaths in a region taking into account the correlation between daily new cases and daily new deaths over time.

## Package Description

The code takes as input a region and the target variable (cases, deaths). Computes the model parameters, the metrics and the prediction.  
* If the target is the cases, the code selects the model.
* If the target is the deaths, the code computes the parameters of a linear or non-linear two-dimensional model.
* If the user requests a prediction of the number of deaths at a given date, using the model for the cases over time, the code predicts the number of deaths (*not implemented yet*).

### Usage
The command line:

    $python covid_19.py -h

returns information about the usage and different options:

    usage: covid_19.py [-h] [-m [{linear,non-linear}]] [-r [REGION]]
                 [-t [{deaths,cases}]]
                 [-o [{knn,LocalOutlierFactor,EllipticEnvelope}]]
                 [-op [OUTLIERS_PORTION]]

    Parse various arguments.

    optional arguments:
    -h, --help            show this help message and exit
    -m [{linear,non-linear}], --model [{linear,non-linear}]
                        Choose statistical model. Default is linear.
    -r [REGION], --region [REGION]
                        Choose country, country code, geoid, continent or all. Default is all.
    -t [{deaths,cases}], --target [{deaths,cases}]
                        Choose the target. Default is deaths.
    -o [{knn,LocalOutlierFactor,EllipticEnvelope}], --outliers_method [{knn,LocalOutlierFactor,EllipticEnvelope}]
                        Choose method for the detection of outliers. Default is knn.
    -op [OUTLIERS_PORTION], --outliers_portion [OUTLIERS_PORTION]
                        Provide the portion of outliers in the dataset. Default is 0.1.

##### Examples:

* To request a model that describes the evolution of COVID-19 cases with time in Italy, you may use the following command line :

      $ python covid_19.py -t cases -r Italy  
  It returns :

      The model is :  Gaussian
      The R squared is : 0.5035300768867241

* To train a linear model for the number of deaths in Italy, you may use the following command line :

      $ python covid_19.py -m linear -r Italy
  It returns :

      The coefficients for [time cases] are :  [1.81235189 0.12551218]
      The value of the regularization parameter is :  1
      The MSE is :  5932.507784901325
      The R squared is :  0.848895108773938
      Saving plots to deaths.png

* To train a non-linear model for the number of deaths in Italy, you may use the following command line :

      $ python covid_19.py -m non-linear -r Italy
  It returns :

      Polynomial regression with least squares :
      The degree of the polynomial is :  3
      The MSE is :  5275.404436654533
      The R squared is :  0.9031394281865164
      Saving plots to deaths.png

### Data

The dataset used in this package contains the number of daily new cases and new deaths due to COVID-19 by country and continent. It is updated on a daily basis.

### Statistical Models

* For the daily new cases the code fits three different models (linear, gaussian and logistic) and selects the model with the R squared score that is closest to 1.

* A linear and non-linear models are supported for mapping daily new deaths to daily new cases in time. The linear model includes regularization with cross-validation in order to optimize the R squared metric. For non-linear regression, the code fits the polynomial that optimizes the R squared metric and includes regularization effects.

### Code Organization

```
COVID-19
├── covid_19.py
├── models
│   ├── prep_data.py
│   ├── model_cases.py
│   └── models.py
├── output_files
│   └── linear.py
└── README.md
```

## Installation


## Acknowledgements

The [dataset](https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide) used in this package has been made publically available by the European Centre for Disease Prevention and Control (ECDC).

## Contributions
