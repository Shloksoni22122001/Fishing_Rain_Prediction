# Fishing Rain Prediction

## Overview

This project aims to predict whether it will rain on a given day based on various meteorological features. The dataset used for this prediction is stored in an SQLite database. The project involves data cleaning, feature engineering, and building machine learning models for rain prediction.


## Authors

- [@Shlok](https://www.github.com/Shloksoni22122001)


## Requirements

Before running the code, ensure you have the following libraries installed...

You can set up the environment by creating a Conda environment using the provided `environment.yml` file. - 'doing this is **recomended**'

All Requirementsare already stated in the environment.yml still mentioning some of them...

- python=3.10.12=h966fe2a_0
- catboost=1.2=py310haa95532_0
- graphviz=2.50.0=hdb8b0d4_0
- ipykernel=6.19.2=py310h9909e9c_0
- lightgbm=4.0.0=py310h00ffb61_0
- matplotlib-base=3.7.1=py310h4ed8f06_1
- matplotlib-inline=0.1.6=py310haa95532_0
- pandas=1.5.3=py310h4ed8f06_0
- plotly=5.9.0=py310haa95532_0
- scikit-learn=1.2.0=py310hd77b12b_0
- scipy=1.11.1=py310h309d312_0
- seaborn=0.12.2=py310haa95532_0
- sqlite=3.41.2=h2bbff1b_0
- numpy==1.24.3
- pip==23.2.1
- tensorflow==2.13.0
- xgboost==1.7.6

[Note: at the end of yml file specify the path of yours]


## Usage

1. Clone the repository to your local machine.
2. Create a Conda environment using the provided `environment.yml` file:
```
conda env create -f environment.yml
```
3. Activate the Conda environment:
```
conda activate fishing_rain_prediction
```
4. Run the Jupyter Notebook Solution.ipynb or python program Solution.py to execute the code and explore the project.
## Project Structure

The project is structured as follows:

- **Solution.ipynb**: Jupyter Notebook containing the code for data cleaning, feature engineering, model building, and evaluation.
- **environment.yml**: Conda environment file specifying the required dependencies.
- **fishing.db**: SQLite database containing the dataset.(which is not provided here as for the size problem...)
- **README.md**: This README file providing an overview of the project.

## Data Cleaning and Feature Engineering

- The project involves cleaning the data, handling missing values, and performing feature engineering to prepare the dataset for modeling.
- Features like date columns are converted into day, month, and year components.
- Categorical data is transformed into numerical ordinal values.
- Missing values are filled using rolling median and mode imputation.
## Data Cleaning and Feature Engineering

- The project involves cleaning the data, handling missing values, and performing feature engineering to prepare the dataset for modeling.
- Features like date columns are converted into day, month, and year components.
- Categorical data is transformed into numerical ordinal values.
- Missing values are filled using rolling median and mode imputation.
## Model Building and Evaluation

- Various machine learning models are trained and evaluated for rain prediction, including Logistic Regression, Decision Trees, Random Forest, XGBoost, LightGBM, CatBoost, Support Vector Classifier, Neural Network, and Naive Bayes (Gaussian and Bernoulli).
- The models are evaluated based on accuracy and other relevant metrics.
- The best-performing model is selected based on the evaluation results.

## Conclusion

The project concludes with the selection of the best-performing model for rain prediction based on the given dataset. The chosen model achieved an accuracy score of approximately 97% using log_loss as the evaluation metric.

Feel free to explore the Jupyter Notebook for detailed code and analysis.
