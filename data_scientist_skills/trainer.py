import joblib
from google.cloud import storage
from termcolor import colored
from data_scientist_skills import model, recommended_jobs

### Implement params.py so that the gcp information doesn't need to be defined
### in every .py -JP
BUCKET_NAME = 'skills_for_a_data_scientist'

###Not needed because of how model is currently invoked -JP
# BUCKET_TRAIN_DATA_PATH = 'data/raw_data/DataAnalyst.csv'
# BUCKET_TRAIN_DATA_PATH_2 = 'data/raw_data/DataScientist.csv'

##### Model storage GCP - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'skills'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'

### - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

STORAGE_LOCATION = f'models/{MODEL_NAME}/{MODEL_VERSION}/model.joblib'

def upload_model_to_gcp():
    """Uploads model.joblib to GCP, currently unused."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STORAGE_LOCATION)
    blob.upload_from_filename('model.joblib')

### This method could be rewritten to handle an arbritary amount of python
### objects -JP
#Take care in case you want to save another model, as written this method
#overwrites any saved model.joblib of the same name
def save_model(reg):
    """method that saves the model(s) into a .joblib file and *could* upload it on Google
    Storage /models folder use joblib library and google-cloud-storage"""

    #saving the trained model locally
    joblib.dump(reg, f'{reg.nickname}_model.joblib')
    print(colored(f"{reg.nickname}_model.joblib' saved locally", "green"))

    # Saves trained model to GCP. Currently unused.
    # upload_model_to_gcp()
    # print(colored(f"uploaded{reg.nickname}_model.joblib' \
        # to gcp cloud storage under \n => {STORAGE_LOCATION}", "blue"))


if __name__ == '__main__':
    # Runs the imported class objects and saves them via save_model method
    ### With restructured code we could have more flexible definitions here
    ### e.g. where/how the data is imported and processed, enabling GCP training-JP

    #Instantiates the model objects
    svc_and_vec_model = model.Model()
    jobs_rec_model = recommended_jobs.Recommend_Jobs()


    #Saves trained models locally (and GCP bucket if lines in save_model
    #are uncommented)
    save_model(svc_and_vec_model)
    save_model(jobs_rec_model)
