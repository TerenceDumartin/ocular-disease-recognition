import os
import imageio
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from google.cloud import storage
import numpy as np
from pathlib import Path

BUCKET_NAME = "ocular-disease-recognition"
DATA_FOLDER = "data/Train_test_split/"

def get_data(local, mode, target_v0, target_v1):
    if not local :
        #chack if files already downloaded in cloud
        if not os.path.exists(DATA_FOLDER+'X_test.csv'):
            download_blob()

    #csv to DataFrame
    df_train = get_df_local('X_train.csv') #for testing
    df_test = get_df_local('X_test.csv') #for testing
    
    if mode == 'v0':
        y_train = df_train[target_v0]
        y_test  = df_train[target_v0]
    else:
        #Remove normal observation + create y classifier
        df_train = df_train[df_train.N==0]
        y_train = df_train[target_v1].copy()
        y_train['O'] = 0
        y_train.loc[y_train.sum(axis=1) == 0,'O'] = 1

        df_test = df_test[df_test.N==0]
        y_test = df_test[target_v1].copy()
        y_test['O'] = 0
        y_test.loc[y_test.sum(axis=1) == 0,'O'] = 1

    #images to X
    X_train = get_images_local(df_train, 'Train')
    X_test = get_images_local(df_test, 'Test')

    return y_train, y_test, X_train, X_test

#------------------------------------------------
#     LOCAL IMPORT
#------------------------------------------------

def get_images_local(df, folder):
    print(f'loading images {folder}')
    img_data = []

    path = f"{DATA_FOLDER}/{folder}/"
    for file in df.filename:
        img_data.append(imageio.imread(path + file))
    return np.array(img_data)

def get_df_local(data_file):
    
    # load n lines from my csv
    print(f'loading df {data_file}')
    path = DATA_FOLDER + data_file
    df = pd.read_csv(path)#, nrows=10) #for debugging
    return df

#------------------------------------------------
#     GLOB IMPORT
#------------------------------------------------

def download_blob():
    
    print(f'DOWNLOADING FROM STORAGE ')
    prefix = 'data/Train_test_split/'
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=prefix)  # Get list of files

    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        file_split = blob.name.split("/")
        directory = "/".join(file_split[0:-1])
        Path(directory).mkdir(parents=True, exist_ok=True)
        blob.download_to_filename( blob.name)
    
def save_model_to_gcp(joblib_name):

    storage_location = f"models/{joblib_name}"
    local_model_filename = joblib_name

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(storage_location)
    blob.upload_from_filename(local_model_filename)

def save_image_to_gcp(fig_name):

    storage_location = f"models/{fig_name}"
    local_model_filename = DATA_FOLDER + '/' + fig_name

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(storage_location)
    blob.upload_from_filename(local_model_filename)
