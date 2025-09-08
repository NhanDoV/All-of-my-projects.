import pandas as pd
import numpy as np
from datetime import datetime
import string

def valid_IP4(address):
    """
    
    """    
    parts = address.split(".")
    if len(parts) != 4:
        return False
    for item in parts:
        if not 0 <= int(item) <= 255:
            return False
    return True

def valid_IP6(address):
    """
    
    """    
    parts = address.split(":")
    if len(parts) == 8:
        for item in parts:
            if len(item) <= 4:
                if all(s in string.hexdigits for s in item):
                    return True
    else:
        return False
    
def ip_type(address):
    """
    
    """    
    if valid_IP4(address):
        return 'IPv4'
    elif valid_IP6(address):
        return 'IPv6'
    elif address == '0':
        return 'unknown_ip'
    else:
        return 'invalid_ip'

def extract_db_calc_difftime(df, ad_id):
    """
    
    """    
    sub_df = df[df['ad_id'] == ad_id][['ad_id', 'utc_datetime', 'distance']]
    N = len(sub_df)
    ellapse = []
    for k in range(N):
        if k+1 < N:
            x = str(sub_df['utc_datetime'].values[k]).replace('T', ' ')[:-10]
            y = str(sub_df['utc_datetime'].values[k+1]).replace('T', ' ')[:-10]
            diff = datetime.strptime(y, "%Y-%m-%d %H:%M:%S") - datetime.strptime(x, "%Y-%m-%d %H:%M:%S") 
            diff = diff.seconds
        else:
            diff = 0
        ellapse.append(diff)
    sub_df['ellapse_in_seconds'] = ellapse
    return sub_df

def create_sample_df(df, ad_id, col_names=['ad_id', 'horizontal_accuracy', 'latitude', 'longitude', 'utc_datetime']):
    """
    
    """
    ## tọa độ nhà hàng
    long_cen, lat_cen = -96.770709, 32.810594
    
    ## Width per 1 degree of latitude & longitude
    ratio_long, ratio_lat = 54.6*1.609, 1.609*69 
    
    ## extract columns
    sub_df = df[df['ad_id'] == ad_id][col_names]
    
    sub_df['Han_dist'] = df[df['ad_id'] == ad_id]['distance']  ## use sphere's assumtion
    
    ## tính lại distance ()
    sub_df['distance'] = np.sqrt(  (ratio_long*(sub_df.longitude.values - (long_cen)))**2
                                 + (ratio_lat*(sub_df.latitude.values - (lat_cen)))**2
                                )
    N = len(sub_df)
    ellapse = []
    diff_distance = []
    for k in range(N):
        if k+1 < N:
            x = str(sub_df['utc_datetime'].values[k]).replace('T', ' ')[:-10]
            y = str(sub_df['utc_datetime'].values[k+1]).replace('T', ' ')[:-10]
            diff = datetime.strptime(y, "%Y-%m-%d %H:%M:%S") - datetime.strptime(x, "%Y-%m-%d %H:%M:%S") 
            diff = diff.seconds
            
            diff_dist = np.sqrt(  (ratio_lat*(sub_df['latitude'].values[k+1] - sub_df['latitude'].values[k]))**2
                                + (ratio_long*(sub_df['longitude'].values[k+1] - sub_df['longitude'].values[k]))**2)
        else:
            diff = 0
            diff_dist = 0
        ellapse.append(diff)
        diff_distance.append(diff_dist)

    ## calculate speed
    speed = []
    for k in range(len(diff_distance)):
        if ellapse[k] == 0:
            ## velocity
            veloc = diff_distance[k] / (ellapse[k] + 1)
        else:
            veloc = diff_distance[k] / ellapse[k]
        speed.append(1000*veloc)
    sub_df['Texas_time'] = sub_df['utc_datetime'].apply(lambda x: str((x.hour - 6)%24) + 'h' + str(x.minute) + 'm' + str(x.second) + 's' ) 
    sub_df['Texas_hour'] = sub_df['utc_datetime'].apply(lambda x: (x.hour - 6)%24 )    
    sub_df['diff_time_sec'] = ellapse
    sub_df['diff_dist_km'] = diff_distance
    sub_df['speed_meter_per_sec'] = speed
    sub_df['day'] = sub_df['utc_datetime'].apply(lambda x: x.day)
    
    sub_df['horizontal_accuracy'] = sub_df['horizontal_accuracy'] / 1000
    sub_df = sub_df.rename(columns = {'horizontal_accuracy': 'horz_acc_km'})
    
    return sub_df

def count_movement(ad_id_df, day, thresh_1 = 60, thresh_2 = 8, thresh_3 = 20, thresh_4 = 7200, err = 0.18):
    """
        Input parameters:
            ad_id_df : dataframe contain ad_id
            day: extracted df by day
            thresh_1: threshold_value of time when a customer stop to move the next step
            thresh_2: thresh_value of time that if there exist a time that have a change of direction_movement
            thresh_3: min_time that the customer visiting a restaurant
            thresh_4: max_time that the customer visiting a restaurant
        Return: Numbers of "change_direction" & "movement_with_stoping_time"
    """
    new_df_day = ad_id_df[ad_id_df['day'] == day]
    distance = new_df_day.distance
    avg_distance_p_day = new_df_day.distance.mean()
    diff_dist = new_df_day.diff_dist_km.values
    diff_time = new_df_day.diff_time_sec.values
    horz_acc = new_df_day.horz_acc_km.values
    
    ## speed: meter per seccond
    speed_ms = new_df_day.speed_meter_per_sec
    
    ## Another params
    total_dist_2_res = 0
    count_change_dir = 0
    count_entered = 0
    count_dist = 0
    
    N = len(diff_dist)
    
    ## only count the day has at least one observation
    if N > 0:
        for k in range(N):            
            if (k < N - 1):
                
                ## this meant that if there exist a time that have a change of direction_movement and longer than 10 seconds
                if (diff_dist[k] * diff_dist[k+1] < 0) & \
                                (diff_time[k] > thresh_2) & (speed_ms[k] < 1.25) \
                                & (speed_ms[k+1] > 6.25):
                    count_change_dir += 1
                
                ## this meant if there exist the relax time that longer than 30 seconds that near the restaurant
                if (diff_time[k] > thresh_1) & (diff_dist[k] != 0) & \
                                (distance[k] <= 0.285) & (speed_ms[k] < 3.25) & \
                                (speed_ms[k+1] > 6.25):
                    count_dist += 1
                               
                ## count entered: 
                ## - diff_time in [thresh_3, thresh_4] 
                ## - speed[k] :  < 0.001 (m/s)
                if (distance[k] <= 0.25) & (speed_ms[k+1] < 1) & \
                                (np.abs(distance[k] - 4*horz_acc[k]) < err) & \
                                (diff_time[k] > thresh_3) & (diff_time[k] < thresh_4):
                    count_entered += 1
                    total_dist_2_res += distance[k]
                
            count_dist = max(max(count_dist, count_change_dir), count_entered)                
    
    if (count_entered != 0):
        avg_distance_2_rest = min(avg_distance_p_day, total_dist_2_res / count_entered)    
    else:
        avg_distance_2_rest = np.nan
        
    return count_change_dir, count_dist, count_entered, avg_distance_2_rest, avg_distance_p_day

def frame_prob(df, ad_id):
    """
    
    """    
    sub_df = create_sample_df(df, ad_id)
    new_df = sub_df.set_index('utc_datetime')

    frame = pd.DataFrame(columns = ['ad_id', 'count_change_dir', 'count_mvm', 'count_entered',
                                    'avg_distance_2_rest', 'avg_distance_p_day', 'prob'])
    for day in range(24, 31):
        count_change_dir, count_dist, count_entered, avg_dist_2_rest, avg_dist_p_day = count_movement(new_df, day)
        if count_dist != 0: 
            prob = count_entered / count_dist
        else:
            prob = 0
        frame.loc[day] = [ad_id, count_change_dir, count_dist, count_entered, avg_dist_2_rest, avg_dist_p_day, prob]

    return frame

def find_time_come_home(data, day = 24):
    """
        This function returns the texas_time that someone apart or depart somewhere (from 0am to 7am)
    """
    home = data[((data['Texas_hour'] >= 0) & (data['Texas_hour'] <= 7))]
    home = home[home.day == day]
    speed = home.speed_meter_per_sec.values
    stay = home.diff_time_sec.values
    time = home.Texas_time.values
    hour = home.Texas_hour.values
    N = len(stay)
    for k in range(N - 3):
        if (speed[k] < 1.2) & (speed[k+1] > 2) & (speed[k+2] < 1.2):
            print(time[k+1])


def comeback_home_coordinate(data, day = 24):
    """
        This function returns the (longitude, latitude) that someone come home (from 0am to 7am)
    """
    count = 0
    home = data[((data['Texas_hour'] >= 0) & (data['Texas_hour'] <= 7))]
    home = home[home.day == day]
    
    X = []; Y = []
    speed = home.speed_meter_per_sec.values
    stay = home.diff_time_sec.values
    time = home.Texas_time.values
    hour = home.Texas_hour.values
    x = home.longitude.values
    y = home.latitude.values
    N = len(stay)
    
    
    for k in range(N - 3):
        if (speed[k] < 1.2) & (speed[k+1] > 2) & (speed[k+2] < 1.2):
            count += 1
            X.append(x[k])
            Y.append(y[k])
    if (count <= 3) & (count > 1):
        X = X[0]
        Y = Y[0]
    else:
        X, Y = np.nan, np.nan
    return X, Y

def home_address_per_day(df, ad_id, addition_info, 
                         col_names = ['ad_id', 'count_change_dir', 'count_mvm', 'count_entered',
                                      'avg_distance_2_rest', 'avg_distance_p_day', 
                                      'prob', 'lati_home', 'logi_home']):
    """
    
    """    
    sub_df = create_sample_df(df, ad_id)
    frame = pd.DataFrame(columns = col_names)    
    for day in range(24, 31):
        x, y = comeback_home_coordinate(sub_df, day)
        count_change_dir, count_dist, count_entered, avg_dist_2_rest, avg_dist_p_day = count_movement(addition_info, day)
        if count_dist != 0: 
            prob = count_entered / count_dist
        else:
            prob = 0
        frame.loc[day] = [ad_id, count_change_dir, count_dist, count_entered, avg_dist_2_rest, avg_dist_p_day, prob, y, x]

    return frame
