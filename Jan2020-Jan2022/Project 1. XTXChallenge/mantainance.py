"""
    Noting that this script only implemented on AWS Lambda
    All the credential info has been stored in the S3 bucket and SSM parameter store
"""
#========================== import libraries ===============================
import boto3, os, requests
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf
import aws_cdk.aws_ssm as ssm

#==================== loading environment variables =========================
s3_bucket1 = os.environ["S3-predict"]
data_dir1 = os.environ["S3-dir-pred"]
s3_bucket2 = os.environ["S3-actual"]
data_dir2 = os.environ["S3-dir-actual"]
#==================== your function here =========================
def get_real_time_data(s3_bucket, data_dir):
    """

    """
    # 's3' is a key word. create connection to S3 using default config and all buckets within S3
    s3 = boto3.client('s3') 
    try:
        # get object and file (key) from bucket
        obj = s3.get_object(Bucket= s3_bucket, Key= data_dir) 
        # 'Body' is also a key word
        df = pd.read_csv(obj['Body'])
    except:
        print("Please check again your bucket-name or the data_dir is right or not?")

    return df

def get_mean_and_std(s3_bucket, data_dir):
    """
        Get mean and std value in the last 6 months of y which stored daily in S3
    """
    s3 = boto3.client('s3')
    obj = s3.list_objects(Bucket = s3_bucket, Prefix = data_dir) 
    for o in obj.get('Contents'):
        data = s3.get_object(Bucket=s3_bucket, Key=o.get('Key'))
        mean, std = data['Body'].read()
    return mean, std

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

def send_message(content):
    """
        Send message to an email
    """
    ses_client = boto3.client("ses", region_name="xxx")
    CHARSET = "UTF-8"

    response = ses_client.send_email(
        Destination={
            "ToAddresses": [
                "xxxxxxx this is credential info xxxx",
            ],
        },
        Message={
            "Body": {
                "Text": {
                    "Charset": CHARSET,
                    "Data": content,
                }
            },
            "Subject": {
                "Charset": CHARSET,
                "Data": "xxxxxxx this is credential info xxxx",
            },
        },
        Source="xxxxxxx this is credential info xxxx",
    )
    print(response)

def get_ssm_parameter_store():
    secure_string_token = ssm.StringParameter.value_for_secure_string_parameter("my-secure-parameter-name")
    x1,x2,x3,x4 = secure_string_token.split("\*")
    return x1, x2, x3, x4

def accuracy_monitoring(s3_bucket, data_dir):
    """
        Find the model accuracy
            If good, keep them
            If not good
                define warning / alert
                if alert we need to remodel
    """
    actual_df = get_real_time_data(s3_bucket2, data_dir2)
    predict_df = get_real_time_data(s3_bucket1, data_dir1)
    y_true = actual_df['y']
    y_pred = predict_df['y_pred']
    
    # These threshold used to determine the models is in-warning or in-alert
    x1, x2, x3, x4 = get_ssm_parameter_store()

    accs = forecast_accuracy_2(y_pred, y_true)

    if (accs['mae'] < x1)*(accs['rmse'] < x2)*(accs['mae'] > x3)*(accs['rmse'] > x4):
        contents = "warning"
    elif (accs['mae'] > x1)*(accs['rmse'] > x2):
        content = 'alert, next to re-model'

    send_message(content)
