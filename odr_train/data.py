
import os
import imageio
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from google.cloud import storage
import numpy as np

BUCKET_NAME = "lwg-ocular-disease-recognition"

def get_data(local):
    if local :
        #csv to DataFrame
        df_train = get_df_local('X_train.csv', nrows = 1) #for testing
        df_test = get_df_local('X_test.csv', nrows = 1) #for testing
        
        #images to X
        X_train = get_images_local(df_train, 'Train')
        X_test = get_images_local(df_test, 'Test')
        
    else :
        #csv to DataFrame
        df_train = get_df_using_blob('X_train.csv')
        df_test = get_df_using_blob('X_test.csv')
        
        # #images to X
        # X_train = get_images_using_blob(df_train, 'Train')
        # X_test = get_images_using_blob(df_test, 'Test')
        X_train=0
        X_test=0
    return df_train, df_test, X_train, X_test

#------------------------------------------------
#     LOCAL IMPORT
#------------------------------------------------

def get_images_local(df, folder):
    print(f'loading images {folder}')
    img_data = []

    path = f"data/Train_test_split/{folder}/"
    for file in df.filename:
        img_data.append(imageio.imread(path + file))
    return np.array(img_data)

def get_df_local(data_file, nrows):
    
    # load n lines from my csv
    print(f'loading df {data_file}')
    path = "data/Train_test_split/" + data_file
    df = pd.read_csv(path, nrows = nrows)
    return df

#------------------------------------------------
#     GLOB IMPORT
#------------------------------------------------

def get_df_using_blob(data_file):

    print(f'loading df {data_file}')
    # get data from my google storage bucket
    BUCKET_TRAIN_DATA_PATH = "data/Train_test_split/" + data_file

    client = storage.Client()  # verifies $GOOGLE_APPLICATION_CREDENTIALS
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(BUCKET_TRAIN_DATA_PATH)

    blob.download_to_filename(data_file)

    # load downloaded data to dataframe
    df = pd.read_csv(data_file)

    return df

def get_images_using_blob(df, folder):
    print(f'loading images {folder}')
    
    client = storage.Client()

    img_data = []
    for filename in df.filename:
        BUCKET_TRAIN_DATA_PATH = "data/Train_test_split/" + folder +'/' + filename
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(BUCKET_TRAIN_DATA_PATH)
        blob.download_to_filename(filename)
        img_data.append(imageio.imread(filename))
    
    return np.array(img_data)


def save_model_to_gcp(joblib_name):

    storage_location = f"models/{joblib_name}"
    local_model_filename = joblib_name

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(storage_location)
    blob.upload_from_filename(local_model_filename)

