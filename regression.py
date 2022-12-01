import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import os

##########################################
# Data Sourcing
##########################################

# read the dataset
df = pd.read_csv('data/auto-mpg.data', lineterminator='\n', header=None)
df[[0, 'car_name']] = df[0].str.split('\t', expand=True)
df.head()

# define column names
columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']

# refine the dataframe
df[columns] = df[0].str.split(expand=True)
df.drop(columns=[0], inplace=True)
df['car_name'] = df['car_name'].apply(lambda x: x.replace('"', ''))
df.head()

# convert colums to float type
for col in df.columns:
    if col not in ['mpg', 'car_name']:
        df = df[pd.to_numeric(df[col], errors='coerce').notnull()]
        df[col] = df[col].astype(float)

# seperate dependant and independant variables
X = df[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']]
y = df['mpg']

# train test split
train_X, test_X, train_y, test_y = train_test_split(X,y,random_state=0)

# function to evaluate model performance
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    return rmse, mae

##########################################
# Using MLFlow
##########################################

# define a new experiment
experiment_name = "PlainRegression"

# return experiment ID
try:
    # create a new experiment
    exp_id = mlflow.create_experiment(name=experiment_name)
except Exception as e:
    exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# create a 'images' directory
if 'images' not in os.listdir():
    os.mkdir('images')

with mlflow.start_run(experiment_id=exp_id):
    # simulate EDA process by creating distribution plots for all the features
    train_X.plot(kind='box', subplots=True, layout=(2,4), figsize=(16,9), title='Box plot of each feature')

    # save the image to images folder
    plt.savefig('images/distribution_plot_all_features.png')

    # log artifacts -> saves the image and enables tracking for later use
    mlflow.log_artifacts('images')

    # define alpha and l1 ratio
    alpha, l1_ratio = 0.02, 0.15

    # initiate an elastic net model
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)

    # fit the model with train dataset
    lr.fit(train_X, train_y)

    # make predictions on test set
    y_pred = lr.predict(test_X)

    # obtain the model performance
    rmse, mae = eval_metrics(test_y, y_pred)

    # log the parameters
    mlflow.log_param('alpha', alpha)
    mlflow.log_param('l1_ratio', l1_ratio)

    # log the metrics
    mlflow.log_metric('rmse', rmse)
    mlflow.log_metric('mae', mae)

    # save the model for later use
    mlflow.sklearn.log_model(lr, "PlainRegressionModel")
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

##########################################
# Hyperparameter Tuning using MLFlow
##########################################

# define a new experiment
experiment_name = "PlainRegressionHyperParameterSearch"
try:
    # create a new experiment
    exp_id = mlflow.create_experiment(name=experiment_name)
except Exception as e:
    exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# define alpha and l1 ratio
alphas, l1_ratios = [0.01, 0.05, 0.1, 0.02, 0.03], [0.15, 0.1, 0.2, 0.25]

for alpha in alphas:
    for l1_ratio in l1_ratios:
        # start a mlflow run, and track them under the experiment defined above
        with mlflow.start_run(experiment_id=exp_id):

            # log artifcats -> saves the images and enables tracking for later use
            mlflow.log_artifacts('images')

            # initiate an elastic net model
            lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)

            # fitting the model with train dataset
            lr.fit(train_X, train_y)

            # make predictions on test set
            y_pred = lr.predict(test_X)

            # obtain the model performance
            rmse, mae = eval_metrics(test_y, y_pred)

            # log hyperparameters defined above
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("l1_ratio", l1_ratio)

            # log performance of the model
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)