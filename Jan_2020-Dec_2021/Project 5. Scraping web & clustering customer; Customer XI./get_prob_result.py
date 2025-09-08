import os
import pandas as pd
from processing import *

# ===========================================================================================
folder_movement_path = {}
x = os.listdir(folder_movement_path)
dims = 0
# ===========================================================================================
for file_name in x:
    df = pd.read_csv(os.path.join(folder_movement_path, file_name))
    print('csv file_name = %s\t|====|\t dimension = %s'%(file_name, df.shape))
    dims += df.shape[0]
print('Total observations: %s'%dims)
# ===========================================================================================
df = pd.read_csv(os.path.join(folder_movement_path, x[-1]))
for file_name in x[:-1] :
    df_x = pd.read_csv(os.path.join(folder_movement_path, file_name))
    df = df.append(df_x)
df['utc_datetime'] = df['utc_timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x))
df['ip_address'] = df['ip_address'].fillna('0')
df['is_ip_valid'] = df['ip_address'].apply(lambda x: ip_type(x))
unique_ad_id = list(df['ad_id'].value_counts().index)
# ===========================================================================================
ad_id = {}
n_day_to_extract = {}
def proba_model_by_adid(ad_id, n_day_to_extract):
    """
    
    """
    sub_df = create_sample_df(df, ad_id)
    new_df = sub_df.set_index('utc_datetime')
    new_df_day = new_df[new_df['day'] == n_day_to_extract]

    frame = frame_prob(new_df_day, unique_ad_id[0])
    for adid in unique_ad_id:
        frame = frame.append(frame_prob(df, adid))

    home = sub_df[((sub_df['Texas_hour'] >= 0) & (sub_df['Texas_hour'] <= 7))]
    home = home[home.day == 24]

    frame = home_address_per_day(df, unique_ad_id[0])
    for adid in unique_ad_id:
        frame = frame.append(home_address_per_day(df, adid, new_df))
    frame.to_csv(f"output/sample_adid_{ad_id}.csv", index=False)
# ===========================================================================================
list_ad_id = df["ad_id"].unique()
for adid in list_ad_id:
    proba_model_by_adid(ad_id, n_day_to_extract)