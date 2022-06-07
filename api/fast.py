from fastapi import FastAPI
import joblib
import nltk
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Union
from enum import Enum

nltk.download('stopwords')

### If the jobs recommender is loaded outside of the get call, this
### model can have its skills attribute match the columns of the
### jobs_recommender model.X columns, as is there's no control on the
### format of the passed dictionary
#Class model for a job classification
class Applicant(BaseModel):
    possessed_skills: dict
    years_of_experience: Union[float, None] = None

#Class for the Job classification model page
class Description(BaseModel):
    description: str
    # mean_salary: Union[float, None] = 0

#class for directing to different model pages
class ModelName(str, Enum):
    description_to_job = "job_model"
    skills_needed = "skills_model"
    salary_offer_estimator = "hr_model"


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

#Dummy test for a greeting
@app.get("/")
def index():
    return {"greeting": "Hello world"}

#Invokes the description classifier model.
@app.post("/models/job_classifier_model")
def take_job_desc(desc : Description):

    #loads model
    pipeline = joblib.load('desc_class_model.joblib')

    #make prediction
    results = pipeline.predict(desc.description)

    #returns job track and level
    return {"job level" : results}

#Invokes the skills classifier model.
@app.post("/models/skills_model")
def take_applicant_info(appl : Applicant):

    #loads model
    pipeline = joblib.load('jobs_rec_model.joblib')

    # make prediction
    results = pipeline.predict(appl.possessed_skills)

    #returns results (pd.DataFrame entries) that have been converted to a json
    #format to be parsed.
    return {"recommendations" : results.to_json()}



# #For if there are multiple models to post from
# @app.get("/models/{model_name}") ### can we use fastapi.HTTPException for missing models?
# def get_model(model_name : ModelName): ###here https://fastapi.tiangolo.com/tutorial/path-params/ they type the param as a ModelName object, but then it breaks on nonexistent calls? Is that better or having some default page?
#     if model_name == ModelName.description_to_job:
#         return {"model_name": model_name, "message": "This can give the job level"}
#     if model_name == ModelName.skills_needed:
#         return {"model_name": model_name, "message": "These are what you need for that job"}
#     if model_name == ModelName.salary_offer_estimator:
#         return {"model_name": model_name, "message": "This is the expected salary for that description"}
#     return {"model_name": model_name, "message": "This isn't a defined model"}
