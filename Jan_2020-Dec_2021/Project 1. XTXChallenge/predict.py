from joblib import load
import pandas as pd

model = load("xgb.joblib")
new_input = {}
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
df = pre_processing(input_dir)
y_mean = df['y'].mean()
y_std = df['y'].mean()
#=================================================================
def normalized(data):
    """
        Normalized your data by subtract the average and 
        divide to the standard-deviation    
    """
    data -= data.mean(axis = 0)
    data /= data.std(axis = 0)
    return data

def get_prediction(data, model):
    """
        Get the prediction and added it into the dataframe
    """
    X = data.drop(columns = 'y')
    X_norm = normalized(X)
    preds = model.predict(X_norm)
    new_data = data.copy()
    new_data['y_pred'] = (preds*y_std + y_mean)
    
    # Noting that we used cumsum in the function `pre-processing` so 
    # we must recover the initial value from this
    # by using .diff
    # the missing data must be filled by the first value when we use cumlative-sum
    df['y_pred'] = df['y_pred'].diff().fillna(df['y_pred'].iloc[0])

    return new_data

get_prediction(new_input, model)