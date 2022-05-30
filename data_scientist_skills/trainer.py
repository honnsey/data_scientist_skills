import joblib
from google.cloud import storage
from termcolor import colored
from data_scientist_skills import data

### Implement params.py so that the gcp stuff isn't defined in every .py

BUCKET_NAME = 'skills_for_a_data_scientist'

###I think this is correct to do both csv
###Not needed because of import?
# BUCKET_TRAIN_DATA_PATH = 'data/raw_data/*.csv'

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'skills'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'

### - - - - - - - - - - - - - - - - - - - - - - - - - - - -


### These need to be tailored to your uses
### Import from our module
### may need to define gscloud_getdata() method

def preprocess(df):
    """method that pre-process the data"""
    pass

### Import from our module
def train_model(x):
    """method that trains the model"""
    print(colored("trained model", "blue"))
    return(x)

### Tailored section close ---------------

STORAGE_LOCATION = f'models/{MODEL_NAME}/{MODEL_VERSION}/model.joblib'

### Implement dynamic model naming for record keeping # Or is that handled in the Makefile command?
### model file type may vary depending on module used
### sklearn - .joblib; tensorflow - /'dir'/*.pb; simple version of model saving.h5
def upload_model_to_gcp():

    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(STORAGE_LOCATION)

    blob.upload_from_filename('model.joblib')


def save_model(reg):
    """method that saves the model into a .joblib file and uploads it on Google
    Storage /models folder use joblib library and google-cloud-storage"""

    #saving the trained model locally
    joblib.dump(reg, 'model.joblib')
    print(colored("model.joblib saved locally", "green"))

    #saving trained model to gcp
    upload_model_to_gcp()
    print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")



if __name__ == '__main__':
    # get training data from GCP bucket ####file path needs to be updated
    df = data.get_data()

    # preprocess data
    process_df = preprocess(df)

    # train model (locally if this file was called through the run_locally command
    # or on GCP if it was called through the gcp_submit_training, in which case
    # this package is uploaded to GCP before being executed)
    reg = train_model("dummy") ### Needs import from our module

    # save trained model to GCP bucket (whether the training occured locally or on GCP)
    save_model(reg)
