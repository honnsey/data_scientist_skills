from fastapi import FastAPI
import joblib
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Union
from enum import Enum

class Applicant(BaseModel):
    possessed_skills: Union[str, None] = None
    salary_mean_offer: Union[float, None] = None
    years_of_experience: Union[float, None] = None

#Class for the Job classification model page
class Description(BaseModel):
    description: str
    # mean_salary: Union[float, None] = 0

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

#Invokes the model.
@app.post("/models/job_model")
def take_job_desc(desc : Description):

    pipeline = joblib.load('model.joblib')

    # make prediction
    results = pipeline.predict(desc.description)

    return {"job level" : results}

#For if there are multiple models to post from
@app.get("/models/{model_name}") ### can we use fastapi.HTTPException for missing models?
def get_model(model_name : ModelName): ###here https://fastapi.tiangolo.com/tutorial/path-params/ they type the param as a ModelName object, but then it breaks on nonexistent calls? Is that better or having some default page?
    if model_name == ModelName.description_to_job:
        return {"model_name": model_name, "message": "This can give the job level"}
    if model_name == ModelName.skills_needed:
        return {"model_name": model_name, "message": "These are what you need for that job"}
    if model_name == ModelName.salary_offer_estimator:
        return {"model_name": model_name, "message": "This is the expected salary for that description"}
    return {"model_name": model_name, "message": "This isn't a defined model"}
