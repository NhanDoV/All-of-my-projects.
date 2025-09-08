import os
import warnings
import numpy as np 
import pandas as pd
import xgboost as xgb
from joblib import dump
from statsmodels.tsa.stattools import acf
from sklearn.model_selection import GridSearchCV
#=================================================================
warnings.filterwarnings("ignore")
#=================================================================
input_dir = r'../input/dataset-for-timeseries/XTX_data.csv'
#=================================================================
def pre_processing(input_dir):
    """
        Load the data as dataframe from input-directory
        Also drop the duplicated values and fill the missing values
    """
    df = pd.read_csv(input_dir)
    df = df.fillna(0)

    ## cumulative sum
    df['y'] = df[['y']].cumsum()

    ## droping duplicated
    df = df.drop_duplicates()

    return df

#=================================================================
def train_test_split(df, p_train = 0.667):
    """
        * Split data (df) in to train and validation sets, also split the targets (y) 
        and the independent-variables (X)
        * We need to split this then normalize this later
    """
    N = len(df)
    y = df.y
    train_size = int(p_train*N)
    X_train = df.iloc[: train_size, :-1]
    y_train = y[: train_size]
    X_test = df.iloc[train_size :, :-1]
    y_test = y[train_size :]
    
    return X_train, y_train, X_test, y_test

#=================================================================
def normalized(data):
    """
        Normalized your data by subtract the average and 
        divide to the standard-deviation
    """
    data -= data.mean(axis = 0)
    data /= data.std(axis = 0)
    return data
#=================================================================

def forecast_accuracy_2(forecast, actual):
    """
        Compute all the metrics evaluation used in regression / time-series
        Input:
            forecast : prediction from your trained-model
            actual : actual data which normalized
    """
    # Mean Error (ME)
    me = np.mean(forecast - actual)  
    
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(forecast - actual))
        
    # Root Mean Squared Error (RMSE)
    rmse = np.mean((forecast - actual)**2)**.5
    
    # correlation
    corr = np.corrcoef(forecast, actual)[0,1]  
        
    # Lag 1 Autocorrelation of Error (ACF1) 
    acf1 = acf(forecast - actual, fft = False)[1] 

    # average of natural ratio
    adj_actual = np.array([y if y !=0 else 0.001 for y in actual])
    avg_rt = np.mean(forecast / adj_actual)
    
    res = ({'me':me, 
            'mae': mae, 
            'rmse':rmse, 
            'acf1':acf1, 
            'corr':corr,
            'avg.rate': avg_rt
            })

    return res

#=================================================================
def training_model(input_dir):
    """
        Train, evaluate then store the model from data at input-directory
    """
    # Loading & split data
    df = pre_processing(input_dir)
    X_train, y_train, X_test, y_test = train_test_split(df, 0.7)

    # Normalized data
    x_train_norm = normalized(X_train.copy())
    y_train_norm = normalized(y_train.copy().ravel())
    x_test_norm = normalized(X_test.copy())
    y_test_norm = normalized(y_test.copy().ravel())

    ## Initialize Regressor vs Hyper-parameters
    xgb_model = xgb.XGBRegressor()
    params = {'n_estimators': [1000, 2000, 5000], 
              'max_depth': [5, 7, 9],
              'learning_rate': [0.3, 0.1, 0.05],
              'min_child_weight': [1, 3, 5]
            }
    clf = GridSearchCV(xgb_model, params)
    clf.fit(x_train_norm, y_train_norm, 
            eval_set=[(x_train_norm, y_train_norm), 
                      (x_test_norm, y_test_norm)], 
            early_stopping_rounds = 50, 
            verbose = 0)
    
    # Select the best-model from GridSearch
    estimator = clf.best_estimator_

    print(100*"-")
    print("Best-params:\n", clf.best_params_)

    # Evaluate model by its predictions
    final_test_pred = clf.predict(x_test_norm)
    final_train_pred = clf.predict(x_train_norm)

    print(100*"=")
    print("Training results:\n", forecast_accuracy_2(final_train_pred, y_train_norm))
    print(100*"-")
    print("Validation results:\n", forecast_accuracy_2(final_test_pred, y_test_norm))
    print(100*"=")

    # Saving models
    dump(estimator, "xgb.joblib")
    print("Model saved successfully!!")
