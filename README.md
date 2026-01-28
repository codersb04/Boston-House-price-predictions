# Boston-House-price-predictions
## Task: 
Predicting the price of the houses in Boston City based on the features.</br>
This comes under supervised learning as we have given labelled dataset.
The problem is a regression problem where the machine needs to predict the outcome in the form of a number, i.e., house price.
## Dataset:
The Boston Housing Dataset is derived from information collected by the U.S. Census Service concerning housing in Boston MA. The following describes the dataset columns: </br></br>

CRIM - per capita crime rate by town</br>
ZN - the proportion of residential land zoned for lots over 25,000 sq. ft.</br>
INDUS - proportion of non-retail business acres per town.</br>
CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)</br>
NOX - nitric oxides concentration (parts per 10 million)</br>
RM - the average number of rooms per dwelling</br>
AGE - the proportion of owner-occupied units built prior to 1940</br>
DIS - weighted distances to five Boston employment centres</br>
RAD - index of accessibility to radial highways</br>
TAX - full-value property-tax rate per $10,000</br>
PTRATIO - pupil-teacher ratio by town</br>
B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town</br>
LSTAT - % lower status of the population</br>
MEDV - Median value of owner-occupied homes in $1000's</br>
The dataset includes 506 different data.</br>

Reference: https://www.kaggle.com/datasets/vikrishnan/boston-house-prices
## Steps to achieve the task:
<b>1. Import Libraries: </b></br>
The dependencies needed for this project are numpy, pandas, matplotlib, sklearn, model_selection, train_test_split, XGBRegressor and metrics.</br></br>
<b>2. Import the dataset:</b></br>
Import the dataset using pandas's read_csv function. Later analysis of the data using shape and describe function</br></br>
<b>3. Understand the correlation between the feature of the dataset: </b></br>
In this, we find the <b>Positive</b> and <b>Negative</b> impact of each feature on another one.</br></br>
<b>4. Splitting the Data and Target: </b></br>
We separate the dataset column-wise by separating the <b>MEDV</b> feature which is the target feature from the remaining set.</br></br>
<b>5. Splitting the data into training and test data:</b></br>
Made use of train_test_split function from sklearn to split the data in row wise such that 20% for testing and the rest for training.</br></br>
<b>6. Model Training:</b></br>
We are using <b>XGBRegressor</b> for building the model. Regression predictive modelling problems involve predicting a numerical value such as a dollar amount or a height. XGBoost can be used directly for regression predictive modelling.</br></br>
<b>7. Model Evaluation:</b></br>
We are performing model evaluation on both the trained data and test data. For checking the model performance we are calculating R squared error and Mean absolute error. Also, build a scatter plot for both evaluations for better visualization between the predicted and actual data.</br>
<b>(a). Prediction and accuracy for trained data:</b></br>
R square error:  0.9999948236320982 </br>
Mean absolute error:  0.0145848437110976 </br>
<b>(b). Prediction and accuracy for test data:</b></br>
R square error:   0.8711660369151691 </br>
Mean absolute error:  2.2834744154238233 </br></br>

Reference: Project 3. House Price Prediction using Machine Learning with Python | Machine Learning Project, Siddhardhan, https://www.youtube.com/watch?v=fw5rkjq4Tfo&list=PLfFghEzKVmjvuSA67LszN1dZ-Dd_pkus6&index=5
