import joblib
from google.cloud import storage
from termcolor import colored
from data_scientist_skills import model

### Implement params.py so that the gcp stuff isn't defined in every .py

BUCKET_NAME = 'skills_for_a_data_scientist'

###Not needed because of how model is invoked
# BUCKET_TRAIN_DATA_PATH = 'data/raw_data/DataAnalyst.csv'
# BUCKET_TRAIN_DATA_PATH_2 = 'data/raw_data/DataScientist.csv'

##### Model storage GCP - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'skills'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'

### - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

STORAGE_LOCATION = f'models/{MODEL_NAME}/{MODEL_VERSION}/model.joblib'

# def upload_model_to_gcp():
#     """Uploads model.joblib to GCP, currently unused."""
#     client = storage.Client()
#     bucket = client.bucket(BUCKET_NAME)
#     blob = bucket.blob(STORAGE_LOCATION)
#     blob.upload_from_filename('model.joblib')

#There is currently no unique naming so any previously saved model is overwritten
#Take care in case you want to save another model
def save_model(reg):
    """method that saves the model into a .joblib file and uploads it on Google
    Storage /models folder use joblib library and google-cloud-storage"""

    #saving the trained model locally
    joblib.dump(reg, 'model.joblib')
    print(colored("model.joblib saved locally", "green"))

    # Saves trained model to GCP. Currently unused.
    # upload_model_to_gcp()
    # print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")



if __name__ == '__main__':
    # get training data from GCP bucket ####file path needs to be updated
    ###With restructured code we could have more flexible definition here
    inst_model = model.Model()

    # # preprocess data
    # step doen in Model instantiation

    # # train model (locally if this file was called through the run_locally command
    # # or on GCP if it was called through the gcp_submit_training, in which case
    # # this package is uploaded to GCP before being executed)
    # step done in Model instantiation

    # save trained model to GCP bucket (whether the training occured locally or on GCP)
    save_model(inst_model)
